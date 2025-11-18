#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Filename: swisstopo_downloader.py
#
# Description:
#   An interactive Streamlit application for discovering, downloading, and
#   processing geospatial data from Swisstopo. Users can define a custom
#   Area of Interest (AOI), find relevant datasets like swissALTI3D and
#   SWISSIMAGE, and generate processed outputs such as merged GeoTIFFs and
#   hillshades.
#
# Author:
#   Florian Denzinger (SLF Davos)
#
# Created:
#   2025-07-03
#
# Last Modified:
#   2025-07-15
#
# Version:
#   1.0
#
# License:
#   MIT
#
# Contact:
#   florian.denzinger@slf.ch
#
# Requirements:
# - Python 3.12+
# - See environment.yml for full details.
# - Key libraries: streamlit, geopandas, rasterio, gdal, folium, pystac-client
#
# Usage:
#   streamlit run swisstopo_downloader.py
#
# ==============================================================================

# --- Standard Library Imports ---
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

# --- Third-Party Imports ---
import folium
import geopandas as gpd
import numpy as np
from osgeo import gdal
import pandas as pd
from pystac_client import Client
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import requests
from shapely.geometry import mapping
import streamlit as st
from streamlit_folium import st_folium

# --- Environment and GDAL Configuration ---
try:
    from pyproj.datadir import get_data_dir

    proj_lib_path = get_data_dir()
    os.environ['PROJ_LIB'] = proj_lib_path
    gdal.UseExceptions()
except ImportError:
    st.error(
        "Could not import from pyproj.datadir. Your environment might be misconfigured."
    )
except Exception as e:
    st.error(f"An error occurred during initial setup: {e}")

# --- Backend Functions ---

# --- swisstopo Functions ---
def get_swisstopo_data_for_aoi(
    aoi_gdf: gpd.GeoDataFrame, collection_id: str, time_range: tuple
) -> list[dict]:
    """Queries the Swisstopo STAC API to find asset URLs and metadata."""
    st.write(f"Querying Swisstopo STAC for collection '{collection_id}'...")
    swisstopo_aoi_wgs84 = aoi_gdf.to_crs(epsg=4326)
    aoi_geom = swisstopo_aoi_wgs84.geometry.union_all()

    stac_url = 'https://data.geo.admin.ch/api/stac/v1'
    catalog = Client.open(stac_url)
    swisstopo_start_date, swisstopo_end_date = time_range
    datetime_str = (
        f"{swisstopo_start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
        f"{swisstopo_end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    search = catalog.search(
        collections=[collection_id],
        intersects=mapping(aoi_geom),
        datetime=datetime_str
    )
    items = search.item_collection()

    assets_info = []
    for item in items:
        found_asset = False
        for asset_key, asset_value in item.assets.items():
            if asset_key.endswith('.tif'):
                asset_info = {
                    'url': asset_value.href,
                    'resolution': item.properties.get('eo:gsd', 'N/A')
                }
                assets_info.append(asset_info)
                found_asset = True
                break
        if not found_asset:
            st.warning(f"No suitable '.tif' asset found for item '{item.id}'.")

    st.write(
        f"Found {len(assets_info)} assets for '{collection_id}' in the "
        "selected time range."
    )
    return assets_info


def _get_data_type(url: str) -> str:
    """Helper function to determine the data type from a URL."""
    if 'swissalti3d' in url:
        return 'swissALTI3D'
    if 'swisssurface3d-raster' in url:
        return 'swissSURFACE3D'
    if 'swissimage-dop10' in url:
        return 'SWISSIMAGE'
    return 'unknown'


def summarize_assets(assets: list[dict]) -> pd.DataFrame:
    """Creates a summary DataFrame of the assets found."""
    if not assets:
        return pd.DataFrame(
            columns=["Data Product", "Year", "Resolution (m)", "File Count"]
        )
    summary_data = []
    year_pattern = re.compile(r'[^/]+_(\d{4})_')
    for asset in assets:
        url, data_type = asset['url'], _get_data_type(asset['url'])
        match = year_pattern.search(url)
        if match and data_type != 'unknown':
            summary_data.append({
                "Data Product": data_type,
                "Year": match.group(1),
                "Resolution (m)": asset['resolution']
            })
    if not summary_data:
        return pd.DataFrame(
            columns=["Data Product", "Year", "Resolution (m)", "File Count"]
        )
    df = pd.DataFrame(summary_data)
    return df.groupby(
        ['Data Product', 'Year', 'Resolution (m)']
    ).size().reset_index(name='File Count')


def _clip_raster(raster_path: Path, aoi_gdf: gpd.GeoDataFrame):
    """Clips a raster to the AOI extent and overwrites the original file."""
    try:
        st.write(f"Clipping {raster_path.name} to AOI...")
        with rasterio.open(raster_path) as src:
            aoi_reprojected = aoi_gdf.to_crs(src.crs)
            out_image, out_transform = mask(
                src, aoi_reprojected.geometry, crop=True
            )
            out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(raster_path, "w", **out_meta) as dest:
            dest.write(out_image)
        st.success(f"Successfully clipped: {raster_path.name}")
    except Exception as clip_error:
        st.error(f"Failed to clip {raster_path.name}. Error: {clip_error}")


def _merge_rasters(
        file_paths: list[Path], output_path: Path, progress_bar
):
    """Helper function to merge multiple raster files.

    Args:
        file_paths (list[Path]): List of paths to input raster files.
        output_path (Path): Path where the merged raster will be saved.
        progress_bar (streamlit.delta_generator.DeltaGenerator): Streamlit
            progress bar object.

    Raises:
        rasterio.errors.RasterioIOError: If writing the merged file fails.
    """
    st.write(f"Merging {len(file_paths)} files into {output_path.name}...")
    sources = [rasterio.open(fp) for fp in file_paths]

    try:
        merged_array, out_transform = merge(sources)
        out_meta = sources[0].meta.copy()

        # Update metadata: BIGTIFF is crucial for files > 4GB
        out_meta.update({
            "driver": "GTiff",
            "height": merged_array.shape[1],
            "width": merged_array.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "BIGTIFF": "YES",  # Fixes the 4GB limit error
            "tiled": True  # Improves read/write performance
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(merged_array)

        st.success(f"Successfully merged and saved to: {output_path}")
        progress_bar.progress(1.0)

    finally:
        # Ensure source files are closed even if the merge fails
        for src in sources:
            src.close()


def process_swisstopo_data(
    working_directory_path: Path, assets: list[dict], file_prefix: str,
    mono_azi: int, multi_azimuths: list[int], process_altitude: int,
    del_downloads: bool, sel_products: list[str],
    clip_to_aoi: bool, aoi_gdf: gpd.GeoDataFrame
):
    """Main processing function for Swisstopo data."""
    urls_by_group = {}
    year_pattern = re.compile(r'[^/]+_(\d{4})_')
    for asset in assets:
        url, data_type = asset['url'], _get_data_type(asset['url'])
        match = year_pattern.search(url)
        if match and data_type != 'unknown':
            year = match.group(1)
            group_key = (data_type, year)
            if group_key not in urls_by_group:
                urls_by_group[group_key] = []
            urls_by_group[group_key].append(url)

    for i, ((data_type, year), group_urls) in enumerate(
        urls_by_group.items()
    ):
        should_process_dem = (
            data_type in ['swissALTI3D', 'swissSURFACE3D']
        ) and any(p.startswith(data_type) for p in sel_products)
        should_process_ortho = (data_type == 'SWISSIMAGE') and any(
            p.startswith(data_type) for p in sel_products
        )
        if not (should_process_dem or should_process_ortho):
            continue

        full_prefix = f"{file_prefix}_{data_type}"
        st.subheader(f"Processing Group: {full_prefix} - Year: {year}")
        year_dir = working_directory_path / f"{full_prefix}_{year}"
        download_dir = year_dir / 'downloaded_tiffs'
        download_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []
        download_bar = st.progress(
            0, text=f"Downloading {len(group_urls)} files..."
        )
        for j, url in enumerate(group_urls):
            try:
                filename = url.split('/')[-1]
                filepath = download_dir / filename
                if not filepath.exists():
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(filepath, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                downloaded_files.append(filepath)
                download_bar.progress(
                    (j + 1) / len(group_urls), text=f"Downloaded {filename}"
                )
            except requests.exceptions.RequestException as download_error:
                st.error(f"Error downloading {url}: {download_error}")
        download_bar.empty()

        if not downloaded_files:
            st.warning("No files were downloaded for this group. Skipping.")
            continue

        merge_bar = st.progress(0, text="Merging files...")
        if data_type == 'SWISSIMAGE':
            merged_ortho_path = year_dir / f"{full_prefix}_ortho_{year}.tif"
            _merge_rasters(downloaded_files, merged_ortho_path, merge_bar)
            if 'SWISSIMAGE Grayscale' in sel_products:
                st.write("Creating greyscale version...")
                greyscale_ortho_path = year_dir / f"{full_prefix}_ortho_greyscale_{year}.tif"
                with rasterio.open(merged_ortho_path) as src:
                    red, green, blue = src.read(1), src.read(2), src.read(3)
                    greyscale = (
                        red * 0.299 + green * 0.587 + blue * 0.114
                    ).astype(rasterio.uint8)
                    meta = src.meta.copy()
                    meta.update(count=1, dtype=rasterio.uint8, compress='lzw')
                    with rasterio.open(
                        greyscale_ortho_path, 'w', **meta
                    ) as dst:
                        dst.write(greyscale, 1)
                st.success(
                    "Successfully created greyscale image: "
                    f"{greyscale_ortho_path}"
                )
                if clip_to_aoi:
                    _clip_raster(greyscale_ortho_path, aoi_gdf)
            if (
                'SWISSIMAGE Color' not in sel_products and
                merged_ortho_path.exists()
            ):
                merged_ortho_path.unlink()
            elif 'SWISSIMAGE Color' in sel_products and clip_to_aoi:
                _clip_raster(merged_ortho_path, aoi_gdf)
        else:  # DEM processing
            merged_dem_path = year_dir / f"{full_prefix}_dem_{year}.tif"
            _merge_rasters(downloaded_files, merged_dem_path, merge_bar)
            if f'{data_type} Hillshade Monodirectional' in sel_products:
                mono_hillshade_path = (
                    year_dir / f"{full_prefix}_hillshade_mono_{year}.tif"
                )
                mono_gdal_options = gdal.DEMProcessingOptions(
                    creationOptions=['COMPRESS=LZW'],
                    azimuth=mono_azi,
                    altitude=process_altitude
                )
                gdal.DEMProcessing(
                    str(mono_hillshade_path), str(merged_dem_path), 'hillshade',
                    options=mono_gdal_options
                )
                st.success(
                    "Monodirectional hillshade saved to: "
                    f"{mono_hillshade_path}"
                )
                if clip_to_aoi:
                    _clip_raster(mono_hillshade_path, aoi_gdf)
            if f'{data_type} Hillshade Multidirectional' in sel_products:
                multi_hillshade_path = (
                    year_dir / f"{full_prefix}_hillshade_multi_{year}.tif"
                )
                temp_hillshade_dir = year_dir / 'temp_hillshades'
                temp_hillshade_dir.mkdir(exist_ok=True)
                try:
                    for azimuth in multi_azimuths:
                        azimuth_hillshade_path = temp_hillshade_dir / f"temp_hillshade_{azimuth}.tif"
                        options = gdal.DEMProcessingOptions(
                            azimuth=azimuth,
                            altitude=process_altitude,
                            creationOptions=['COMPRESS=LZW']
                        )
                        gdal.DEMProcessing(
                            str(azimuth_hillshade_path), str(merged_dem_path), 'hillshade',
                            options=options
                        )
                    hillshade_arrays = [
                        rasterio.open(p).read(1)
                        for p in temp_hillshade_dir.glob("*.tif")
                    ]
                    combined_array = np.mean(hillshade_arrays, axis=0).astype(
                        rasterio.uint8
                    )
                    with rasterio.open(
                        next(temp_hillshade_dir.glob("*.tif"))
                    ) as src:
                        multi_meta = src.meta.copy()
                    with rasterio.open(
                        multi_hillshade_path, "w", **multi_meta
                    ) as dest:
                        dest.write(combined_array, 1)
                    st.success(
                        "Multidirectional hillshade saved to: "
                        f"{multi_hillshade_path}"
                    )
                    if clip_to_aoi:
                        _clip_raster(multi_hillshade_path, aoi_gdf)
                finally:
                    shutil.rmtree(temp_hillshade_dir)
            if not any(
                p.startswith(data_type) and ('DTM' in p or 'DSM' in p)
                for p in sel_products
            ) and merged_dem_path.exists():
                merged_dem_path.unlink()
            elif any(
                p.startswith(data_type) and ('DTM' in p or 'DSM' in p)
                for p in sel_products
            ) and clip_to_aoi:
                _clip_raster(merged_dem_path, aoi_gdf)
        if del_downloads:
            st.write(f"Deleting download folder: {download_dir}")
            shutil.rmtree(download_dir)
            st.success("Successfully deleted download folder.")


def create_map_with_layers(location, zoom, aoi_gdf=None):
    """Creates a Folium map with multiple basemap layers and an optional AOI."""
    folium_map = folium.Map(location=location, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg",
        attr="swisstopo",
        name="Landeskarte farbig",
        show=True
    ).add_to(folium_map)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
        attr="swisstopo",
        name="SWISSIMAGE",
        show=False
    ).add_to(folium_map)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissalti3d-reliefschattierung/default/current/3857/{z}/{x}/{y}.png",
        attr="swisstopo",
        name="swissALTI3D Hillshade (Mono)",
        show=False
    ).add_to(folium_map)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissalti3d-reliefschattierung-multidirektional/default/current/3857/{z}/{x}/{y}.png",
        attr="swisstopo",
        name="swissALTI3D Hillshade (Multi)",
        show=False
    ).add_to(folium_map)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swisssurface3d-reliefschattierung/default/current/3857/{z}/{x}/{y}.png",
        attr="swisstopo",
        name="swissSURFACE3D Hillshade (Mono)",
        show=False
    ).add_to(folium_map)
    folium.TileLayer(
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swisssurface3d-reliefschattierung-multidirektional/default/current/3857/{z}/{x}/{y}.png",
        attr="swisstopo",
        name="swissSURFACE3D Hillshade (Multi)",
        show=False
    ).add_to(folium_map)
    if aoi_gdf is not None:
        folium.GeoJson(aoi_gdf, name="Area of Interest").add_to(folium_map)
    return folium_map


# --- Streamlit UI ---
st.set_page_config(layout="wide")

# --- Custom Title ---
LOGO_IMG_PATH = (
    Path(__file__).resolve().parent / "docs" / "markdown" / "assets" / "logo.png"
)

try:
    col1, col2 = st.columns([0.5, 9], vertical_alignment="center")
    with col1:
        try:
            st.image(str(LOGO_IMG_PATH), width=60)
        except Exception as img_load_error:
            print(f"Error loading image {LOGO_IMG_PATH}: {img_load_error}")
            st.caption(" L ")
    with col2:
        st.title("SLF Data Downloader")
except Exception as title_error:
    print(f"Error setting title: {title_error}")
    st.title("SLF Data Downloader")

if 'aoi_gdf' not in st.session_state:
    st.session_state.aoi_gdf = None
if 'assets_to_process' not in st.session_state:
    st.session_state.assets_to_process = []

# --- Shared AOI Definition ---
st.subheader("1. Define Area of Interest (AOI)")
aoi_method = st.radio(
    "Choose AOI method:", ["Draw on Map", "Upload a File"], horizontal=True
)
if aoi_method == "Upload a File":
    uploaded_file = st.file_uploader(
        "Upload GeoPackage (.gpkg), GeoJSON (.geojson), or zipped Shapefile (.zip)",
        type=['gpkg', 'geojson', 'zip']
    )
    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if temp_path.suffix == '.zip':
                    shutil.unpack_archive(temp_path, temp_dir)
                    shp_files = list(Path(temp_dir).glob('*.shp'))
                    if not shp_files:
                        st.error("No .shp file found in the uploaded zip archive.")
                        st.stop()
                    st.session_state.aoi_gdf = gpd.read_file(shp_files[0])
                else:
                    st.session_state.aoi_gdf = gpd.read_file(temp_path)
            st.success(f"Successfully loaded AOI from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.aoi_gdf = None

st.markdown("#### Map View")
map_center, map_zoom = [46.8, 8.2], 8
if st.session_state.aoi_gdf is not None:
    try:
        aoi_wgs84 = st.session_state.aoi_gdf.to_crs(epsg=4326)
        centroid = aoi_wgs84.geometry.union_all().centroid
        map_center, map_zoom = [centroid.y, centroid.x], 12
    except Exception as e:
        st.warning(
            "Could not calculate AOI center. Using default map view. "
            f"Error: {e}"
        )

m = create_map_with_layers(
    location=map_center, zoom=map_zoom, aoi_gdf=st.session_state.aoi_gdf
)
if aoi_method == "Draw on Map":
    folium.plugins.Draw(
        export=True,
        filename='data.geojson',
        position='topleft',
        draw_options={
            'polyline': False, 'marker': False, 'circlemarker': False,
            'circle': False
        }
    ).add_to(m)
    st.write(
        "Draw a rectangle or polygon on the map. The last drawn shape will be "
        "used as the AOI."
    )
folium.LayerControl().add_to(m)
map_data = st_folium(m, width="100%", height=600)

if aoi_method == "Draw on Map" and map_data and map_data.get(
    "last_active_drawing"
):
    drawn_geom = map_data["last_active_drawing"]
    if drawn_geom:
        st.session_state.aoi_gdf = gpd.GeoDataFrame.from_features(
            [drawn_geom], crs="EPSG:4326"
        )
        st.success("AOI captured from map drawing. Rerunning to update view...")
        st.rerun()

st.markdown("---")

# --- Swisstopo Downloader ---
if st.session_state.aoi_gdf is not None:
    st.header("🇨🇭 Swisstopo Data Processing")
    st.subheader("2. Discover Swisstopo Data")
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input(
            "Start date",
            datetime(2010, 1, 1),
            help="The earliest date for the data search.",
            key="swisstopo_start"
        )
    with date_col2:
        end_date = st.date_input(
            "End date",
            datetime.now(),
            help="The latest date for the data search.",
            key="swisstopo_end"
        )
    if st.button("Discover Swisstopo Data", type="primary"):
        collections = {
            "swissALTI3D": "ch.swisstopo.swissalti3d",
            "swissSURFACE3D": "ch.swisstopo.swisssurface3d-raster",
            "SWISSIMAGE": "ch.swisstopo.swissimage-dop10"
        }
        with st.spinner("Discovering data..."):
            all_assets = []
            for cid in collections.values():
                try:
                    all_assets.extend(get_swisstopo_data_for_aoi(
                        st.session_state.aoi_gdf, cid, (start_date, end_date)
                    ))
                except Exception as e:
                    st.error(f"Failed to query collection {cid}: {e}")
            st.session_state.assets_to_process = all_assets
            if all_assets:
                st.success(
                    "Discovery complete! Found a total of "
                    f"{len(all_assets)} files."
                )
            else:
                st.warning(
                    "No data found for the selected products in the given "
                    "AOI and time range."
                )

    if st.session_state.assets_to_process:
        st.markdown("---")

        if st.session_state.assets_to_process:
            st.subheader("3. Configure and Process Swisstopo Data")

            # --- Year Selection Logic ---
            available_years = set()
            year_pattern_ui = re.compile(r'[^/]+_(\d{4})_')

            for asset in st.session_state.assets_to_process:
                match = year_pattern_ui.search(asset['url'])
                if match:
                    available_years.add(match.group(1))

            sorted_years = sorted(list(available_years), reverse=True)

            selected_years = st.multiselect(
                "Select Years to Process",
                options=sorted_years,
                default=sorted_years,
                help="Select specific years to download and process."
            )

            # Filter assets based on selection
            filtered_assets = []
            for asset in st.session_state.assets_to_process:
                match = year_pattern_ui.search(asset['url'])
                if match:
                    if match.group(1) in selected_years:
                        filtered_assets.append(asset)
                else:
                    # Keep assets where no year pattern is found to be safe
                    filtered_assets.append(asset)

            st.dataframe(summarize_assets(filtered_assets))

            project_prefix = st.text_input(
                "Project Prefix", "MyProject", help="A name for your project.",
                key="swisstopo_prefix"
            )

        output_dir = st.text_input(
            "Output Directory",
            str(Path.cwd() / "geotiff_output"),
            help="The local folder where all output will be saved.",
            key="swisstopo_dir"
        )

        selected_products = []
        st.markdown("**Desired Output Products**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**swissALTI3D**")
            if st.checkbox(
                    "swissALTI3D DTM", value=False, key="alti_dtm"
            ):
                selected_products.append("swissALTI3D DTM")
            if st.checkbox(
                    "swissALTI3D Hillshade Monodirectional", value=False,
                    key="alti_mono"
            ):
                selected_products.append(
                    "swissALTI3D Hillshade Monodirectional"
                )
            if st.checkbox(
                    "swissALTI3D Hillshade Multidirectional", value=False,
                    key="alti_multi"
            ):
                selected_products.append(
                    "swissALTI3D Hillshade Multidirectional"
                )
        with c2:
            st.markdown("**swissSURFACE3D**")
            if st.checkbox(
                    "swissSURFACE3D Raster DSM", value=False, key="surf_dsm"
            ):
                selected_products.append("swissSURFACE3D Raster DSM")
            if st.checkbox(
                    "swissSURFACE3D Hillshade Monodirectional", value=False,
                    key="surf_mono"
            ):
                selected_products.append(
                    "swissSURFACE3D Hillshade Monodirectional"
                )
            if st.checkbox(
                    "swissSURFACE3D Hillshade Multidirectional", value=False,
                    key="surf_multi"
            ):
                selected_products.append(
                    "swissSURFACE3D Hillshade Multidirectional"
                )
        with c3:
            st.markdown("**SWISSIMAGE**")
            if st.checkbox(
                    "SWISSIMAGE Color", value=False, key="img_color"
            ):
                selected_products.append("SWISSIMAGE Color")
            if st.checkbox(
                    "SWISSIMAGE Grayscale", value=False, key="img_gray"
            ):
                selected_products.append("SWISSIMAGE Grayscale")

        with st.expander("Advanced Processing Options"):
            st.markdown("**Hillshade Generation**")
            altitude = st.slider(
                "Altitude (hAl)",
                15,
                90,
                45,
                help="The angle of the light source above the horizon (in degrees)."
            )
            mono_azimuth = st.number_input(
                "Monodirectional Azimuth (hAz)",
                0,
                360,
                315,
                help="The compass direction of the light source for the standard hillshade."
            )
            st.write("Multidirectional Azimuths (hAz):")
            az_col1, az_col2, az_col3 = st.columns(3)
            with az_col1:
                start_az = st.number_input("Start Angle", 0, 360, 0, 45)
            with az_col2:
                stop_az = st.number_input("Stop Angle", 0, 360, 360, 45)
            with az_col3:
                step_az = st.number_input("Step", 1, 90, 45)
            st.markdown("**File Handling**")
            clip_aoi = st.checkbox(
                "Clip final products to AOI extent",
                value=True,
                help=(
                    "If checked, all final output rasters will be clipped to "
                    "the exact boundary of your AOI."
                )
            )
            delete_downloads = st.checkbox(
                "Delete raw downloaded files after processing",
                value=True,
                help=(
                    "If checked, the individual downloaded tiles will be "
                    "deleted to save space, keeping only the final processed files."
                )
            )

        if st.button("Download and Process Swisstopo Data", type="primary"):
            with st.spinner("Processing all data groups..."):
                process_swisstopo_data(
                    Path(output_dir),
                    filtered_assets,
                    project_prefix,
                    mono_azimuth,
                    list(range(start_az, stop_az + 1, step_az)),
                    altitude,
                    delete_downloads,
                    selected_products,
                    clip_aoi,
                    st.session_state.aoi_gdf
                )
            st.snow()
            st.header("✅ Swisstopo workflow completed successfully!")
else:
    st.info("Please define an AOI on the map or by uploading a file to begin.")

# --- Final Footer ---
st.markdown("---")
st.caption("""
    **SLF Data Downloader v1.0**
    Developed by Florian Denzinger (SLF Davos).
    Contact: florian.denzinger@slf.ch

    This Streamlit application provides an interactive interface for downloading and processing Swisstopo data.
    """)