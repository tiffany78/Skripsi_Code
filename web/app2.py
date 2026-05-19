#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:26:19 2026

@author: tipanoii
"""

from pathlib import Path
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from PIL import Image
import math


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Web Visualisasi Zona Potensial",
    layout="wide"
)

st.title("Web Visualisasi Pemetaan Potensi Perairan")


# =====================================================
# PATH UTAMA
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

SEAWEED_DIR = BASE_DIR / "geojsonSeaweed"
REEF_DIR = BASE_DIR / "geojsonReef"
DEPTH_PNG_DIR = BASE_DIR / "depth_png"

DEPTH_TIF_DIR = Path("/Users/tipanoii/doc/TA/code/depth")
DEPTH_OVERLAY_CACHE_DIR = BASE_DIR / "cache_depth_overlay"
DEPTH_OVERLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RGB_LAYER_DIR = BASE_DIR / "rgb_layer"
RGB_OVERLAY_CACHE_DIR = BASE_DIR / "cache_rgb_overlay"
RGB_OVERLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# ATRIBUT POPUP
# =====================================================
POPUP_FIELDS = [
    "zone_id",
    "area_ha",
    "salinity_mean",
    "depth_mean",
    "temperature_mean",
    "sedimentation_mean"
]

POPUP_ALIASES = [
    "ID Zona",
    "Luas Zona (ha)",
    "Rata-rata Salinitas",
    "Rata-rata Kedalaman",
    "Rata-rata Suhu",
    "Rata-rata Sedimentasi"
]


# =====================================================
# FUNGSI UMUM
# =====================================================
def list_geojson_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(folder.glob("*.geojson"))


@st.cache_data
def load_geojson(file_path: str):
    gdf = gpd.read_file(file_path)

    if gdf.empty:
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def validate_popup_columns(gdf, file_name):
    missing = [col for col in POPUP_FIELDS if col not in gdf.columns]

    if missing:
        st.error(
            f"File `{file_name}` tidak memiliki kolom berikut:\n\n"
            + ", ".join(missing)
        )
        return False

    return True


def get_expanded_bounds(gdf, padding_ratio=0.05):
    """
    Menghasilkan bounds dengan sedikit padding.
    Format output:
    [[south, west], [north, east]]
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    width = maxx - minx
    height = maxy - miny

    if width == 0:
        width = 0.01
    if height == 0:
        height = 0.01

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    west = minx - pad_x
    east = maxx + pad_x
    south = miny - pad_y
    north = maxy + pad_y

    return [[south, west], [north, east]]


def create_restricted_map(gdf, min_zoom=9.2):
    """
    Membuat peta yang dibatasi berdasarkan bounds data.
    Pengguna tidak bisa menggeser peta terlalu jauh keluar area data.
    """
    bounds = get_expanded_bounds(gdf)
    south, west = bounds[0]
    north, east = bounds[1]

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=min_zoom,
        tiles="CartoDB positron",

        # Membatasi area peta
        max_bounds=True,
        min_lat=south,
        max_lat=north,
        min_lon=west,
        max_lon=east,

        # Membatasi zoom-out
        min_zoom=min_zoom,
        max_zoom=18,
        control_scale=True
    )

    def style_function(feature):
        return {
            "fillColor": "#2ca25f",
            "color": "#ffffff",
            "weight": 1,
            "fillOpacity": 0.65
        }

    def highlight_function(feature):
        return {
            "fillColor": "#ffcc00",
            "color": "#000000",
            "weight": 3,
            "fillOpacity": 0.85
        }

    popup = folium.GeoJsonPopup(
        fields=POPUP_FIELDS,
        aliases=POPUP_ALIASES,
        localize=True,
        labels=True,
        max_width=450
    )

    tooltip = folium.GeoJsonTooltip(
        fields=["zone_id", "area_ha"],
        aliases=["ID Zona", "Luas (ha)"],
        localize=True,
        sticky=True
    )

    folium.GeoJson(
        data=gdf.to_json(),
        name="Zona Potensial",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=tooltip,
        popup=popup
    ).add_to(m)

    m.fit_bounds(bounds)
    folium.LayerControl().add_to(m)

    return m

# =====================================================
# HALAMAN RUMPUT LAUT DAN TERUMBU 
# =====================================================

def render_two_maps_page(title: str, folder: Path):
    st.header(title)

    files = list_geojson_files(folder)

    if not folder.exists():
        st.error(f"Folder tidak ditemukan:\n\n{folder}")
        return

    if not files:
        st.warning(f"Tidak ada file GeoJSON pada folder:\n\n{folder}")
        return

    file_options = {file.name: file for file in files}
    file_names = list(file_options.keys())

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Peta Kiri")

        selected_left_name = st.selectbox(
            "Pilih GeoJSON peta kiri",
            options=file_names,
            index=0,
            key=f"{title}_left"
        )

        gdf_left = load_geojson(str(file_options[selected_left_name]))

        if gdf_left.empty:
            st.warning("GeoJSON kiri tidak memiliki data zona.")
        elif validate_popup_columns(gdf_left, selected_left_name):
            left_map = create_restricted_map(gdf_left, min_zoom=9.2)
            st_folium(left_map, width=None, height=650, key=f"{title}_left_map")

            with st.expander("Tabel ringkasan peta kiri"):
                st.dataframe(
                    gdf_left[POPUP_FIELDS],
                    use_container_width=True
                )

    with right_col:
        st.subheader("Peta Kanan")

        default_index = 1 if len(file_names) > 1 else 0

        selected_right_name = st.selectbox(
            "Pilih GeoJSON peta kanan",
            options=file_names,
            index=default_index,
            key=f"{title}_right"
        )

        gdf_right = load_geojson(str(file_options[selected_right_name]))

        if gdf_right.empty:
            st.warning("GeoJSON kanan tidak memiliki data zona.")
        elif validate_popup_columns(gdf_right, selected_right_name):
            right_map = create_restricted_map(gdf_right, min_zoom=9.2)
            st_folium(right_map, width=None, height=650, key=f"{title}_right_map")

            with st.expander("Tabel ringkasan peta kanan"):
                st.dataframe(
                    gdf_right[POPUP_FIELDS],
                    use_container_width=True
                )

# =====================================================
# FUNGSI HALAMAN DEPTH
# =====================================================
def get_depth_tif_path(depth_choice, year):
    """
    Contoh:
    depth_choice = "10m"
    year = 2023
    hasil = Depth_10m_2023.tif
    """
    tif_path = DEPTH_TIF_DIR / f"Depth_{depth_choice}_{year}.tif"

    if not tif_path.exists():
        return None

    return tif_path


def get_max_depth(depth_choice):
    if depth_choice == "10m":
        return 10
    elif depth_choice == "35m":
        return 35
    else:
        raise ValueError(f"Pilihan kedalaman tidak dikenali: {depth_choice}")


def get_depth_bounds_latlon(tif_path):
    """
    Mengambil bounds GeoTIFF dalam EPSG:4326.
    Output:
    [[south, west], [north, east]]
    """
    with rasterio.open(tif_path) as src:
        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21
        )

    return [[south, west], [north, east]]


def create_depth_overlay_png(tif_path, depth_choice, year):
    """
    Membuat PNG overlay dari GeoTIFF kedalaman.
    PNG ini hanya untuk visualisasi peta, bukan untuk analisis.
    Nilai analisis tetap dibaca dari GeoTIFF asli saat user klik peta.
    """
    max_depth = get_max_depth(depth_choice)

    output_png = DEPTH_OVERLAY_CACHE_DIR / f"overlay_depth_{depth_choice}_{year}.png"

    # Jika PNG cache sudah ada, tidak perlu dibuat ulang
    if output_png.exists():
        return output_png

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    valid_mask = np.isfinite(arr)

    if nodata is not None:
        valid_mask = valid_mask & (arr != nodata)

    # Jika nilai kedalaman negatif, ubah menjadi positif
    depth_positive = np.where(arr < 0, -arr, arr)

    valid_mask = (
        valid_mask
        & np.isfinite(depth_positive)
        & (depth_positive >= 0)
        & (depth_positive <= max_depth)
    )

    # Normalisasi 0 sampai 1
    norm = np.zeros_like(depth_positive, dtype=float)
    norm[valid_mask] = depth_positive[valid_mask] / max_depth

    # Buat RGBA sederhana
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    # Warna biru semakin gelap untuk kedalaman lebih besar
    rgba[..., 0] = (180 - 120 * norm).clip(0, 255).astype(np.uint8)
    rgba[..., 1] = (220 - 150 * norm).clip(0, 255).astype(np.uint8)
    rgba[..., 2] = 255

    # Transparansi
    rgba[..., 3] = 0
    rgba[valid_mask, 3] = 180

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_png)

    return output_png


def sample_depth_value(tif_path, lat, lon):
    """
    Membaca nilai kedalaman dari GeoTIFF berdasarkan titik klik user.
    Input lat/lon dari peta adalah EPSG:4326.
    """
    with rasterio.open(tif_path) as src:
        # Ubah koordinat dari EPSG:4326 ke CRS raster
        if src.crs.to_string() != "EPSG:4326":
            xs, ys = transform(
                "EPSG:4326",
                src.crs,
                [lon],
                [lat]
            )
            x = xs[0]
            y = ys[0]
        else:
            x = lon
            y = lat

        try:
            row, col = src.index(x, y)
        except Exception:
            return None

        # Cek apakah klik berada di dalam raster
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None

        value = src.read(1, window=Window(col, row, 1, 1))[0, 0]

        if src.nodata is not None and value == src.nodata:
            return None

        if not np.isfinite(value):
            return None

        depth = abs(float(value))

        return depth
    

def get_depth_png_path(depth_choice, year):
    """
    Mencari PNG hasil visualisasi kedalaman.

    Mendukung dua kemungkinan struktur:
    1. depth_png/Depth_10m_2023.png
    2. depth_png/10m/Depth_10m_2023.png
    """

    candidates = []

    # Struktur flat: depth_png/Depth_10m_2023.png
    candidates.extend(sorted(DEPTH_PNG_DIR.glob(f"*{depth_choice}*{year}*.png")))

    # Struktur subfolder: depth_png/10m/Depth_10m_2023.png
    subfolder = DEPTH_PNG_DIR / depth_choice
    if subfolder.exists():
        candidates.extend(sorted(subfolder.glob(f"*{year}*.png")))

    if not candidates:
        return None

    return candidates[0]

# =====================================================
# LAYER RGB
# =====================================================

def get_rgb_tif_path(year):
    """
    Mengambil file RGB berdasarkan tahun.
    Contoh:
    year = 2023
    file = rgb_2023.tif
    """
    rgb_path = RGB_LAYER_DIR / f"rgb_{year}.tif"

    if not rgb_path.exists():
        return None

    return rgb_path


def stretch_band_to_uint8(band, valid_mask):
    """
    Mengubah nilai band menjadi 0-255 agar bisa ditampilkan sebagai PNG.
    Menggunakan percentile stretch supaya citra tidak terlalu gelap/terang.
    """
    valid_values = band[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values)]

    if valid_values.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    p2, p98 = np.nanpercentile(valid_values, [2, 98])

    if p98 - p2 <= 0:
        return np.zeros_like(band, dtype=np.uint8)

    stretched = (band - p2) / (p98 - p2) * 255
    stretched = np.clip(stretched, 0, 255)

    return stretched.astype(np.uint8)


def create_rgb_overlay_png(rgb_tif_path, year, max_size=2000):
    """
    Membuat PNG overlay dari GeoTIFF RGB.
    PNG ini dipakai sebagai layer belakang pada Folium.

    Output:
    - path PNG cache
    - bounds dalam format [[south, west], [north, east]]
    """
    output_png = RGB_OVERLAY_CACHE_DIR / f"overlay_rgb_{year}.png"

    with rasterio.open(rgb_tif_path) as src:
        if src.crs is None:
            raise ValueError(f"GeoTIFF RGB tidak memiliki CRS: {rgb_tif_path}")

        # Reproject virtual ke EPSG:4326 agar cocok dengan Folium
        with WarpedVRT(
            src,
            crs="EPSG:4326",
            resampling=Resampling.bilinear
        ) as vrt:

            bounds = [
                [vrt.bounds.bottom, vrt.bounds.left],
                [vrt.bounds.top, vrt.bounds.right]
            ]

            # Jika PNG cache sudah ada, tidak perlu dibuat ulang
            if output_png.exists():
                return output_png, bounds

            # Batasi ukuran output PNG agar tidak terlalu berat
            scale = min(1.0, max_size / max(vrt.width, vrt.height))
            out_width = max(1, int(vrt.width * scale))
            out_height = max(1, int(vrt.height * scale))

            # Asumsi band 1,2,3 adalah R,G,B
            if vrt.count >= 3:
                indexes = [1, 2, 3]
            else:
                # Jika hanya 1 band, tampilkan sebagai grayscale RGB
                indexes = [1, 1, 1]

            rgb = vrt.read(
                indexes,
                out_shape=(3, out_height, out_width),
                resampling=Resampling.bilinear,
                masked=True
            )

    # Ubah masked array menjadi array float
    mask_array = np.ma.getmaskarray(rgb)

    if mask_array.ndim == 0:
        valid_mask = np.ones((rgb.shape[1], rgb.shape[2]), dtype=bool)
    else:
        valid_mask = ~np.any(mask_array, axis=0)

    rgb_data = rgb.filled(np.nan).astype(float)

    r = stretch_band_to_uint8(rgb_data[0], valid_mask)
    g = stretch_band_to_uint8(rgb_data[1], valid_mask)
    b = stretch_band_to_uint8(rgb_data[2], valid_mask)

    rgba = np.zeros((rgb_data.shape[1], rgb_data.shape[2], 4), dtype=np.uint8)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = 0

    # Area valid dibuat terlihat penuh
    rgba[valid_mask, 3] = 255

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_png)

    return output_png, bounds

# =====================================================
# HALAMAN DEPTH
# =====================================================

def render_depth_page():
    st.header("Pemetaan Kedalaman Air")

    # =====================================================
    # PILIHAN KEDALAMAN DAN TAHUN
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        depth_choice = st.selectbox(
            "Pilih batas kedalaman",
            options=["10m", "35m"],
            index=0
        )

    with col2:
        year = st.selectbox(
            "Pilih tahun",
            options=[2023, 2024, 2025],
            index=2
        )
    
    col3, col4 = st.columns(2)

    with col3:
        rgb_opacity = st.slider(
            "Opacity layer RGB",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05
        )
    
    with col4:
        depth_opacity = st.slider(
            "Opacity layer kedalaman",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05
        )

    # =====================================================
    # AMBIL FILE GEOTIFF
    # =====================================================
    tif_path = get_depth_tif_path(depth_choice, year)

    if tif_path is None:
        st.warning(
            f"File GeoTIFF tidak ditemukan untuk kedalaman {depth_choice} tahun {year}."
        )
        return

    # =====================================================
    # TAMPILKAN PNG DI ATAS PETA
    # =====================================================
    png_path = get_depth_png_path(depth_choice, year)

    st.subheader("Hasil Visualisasi Kedalaman Air")

    if png_path is not None:
        st.write(f"Menampilkan PNG: `{png_path.name}`")
        st.image(
            str(png_path),
            use_container_width=True
        )
    else:
        st.info(
            f"PNG untuk kedalaman {depth_choice} tahun {year} belum ditemukan "
            f"pada folder `{DEPTH_PNG_DIR}`."
        )

    # =====================================================
    # BOUNDS PETA
    # =====================================================
    bounds = get_depth_bounds_latlon(tif_path)
    south, west = bounds[0]
    north, east = bounds[1]

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # =====================================================
    # BUAT OVERLAY PNG DARI GEOTIFF
    # =====================================================
    overlay_png = create_depth_overlay_png(
        tif_path=tif_path,
        depth_choice=depth_choice,
        year=year
    )

    # =====================================================
    # RESET TITIK KLIK JIKA FILE BERUBAH
    # =====================================================
    current_depth_key = f"{depth_choice}_{year}"

    if st.session_state.get("depth_clicked_key") != current_depth_key:
        st.session_state["depth_clicked_key"] = current_depth_key
        st.session_state["depth_clicked_point"] = None

    clicked_point = st.session_state.get("depth_clicked_point")

    # =====================================================
    # MEMBUAT PETA
    # =====================================================
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9.2,
        tiles="CartoDB positron",
        max_bounds=True,
        min_lat=south,
        max_lat=north,
        min_lon=west,
        max_lon=east,
        min_zoom=9.2,
        max_zoom=18,
        control_scale=True
    )

    # =====================================================
    # LAYER RGB SEBAGAI BACKGROUND
    # =====================================================
    rgb_tif_path = get_rgb_tif_path(year)

    if rgb_tif_path is not None:
        rgb_overlay_png, rgb_bounds = create_rgb_overlay_png(
            rgb_tif_path=rgb_tif_path,
            year=year
        )

        folium.raster_layers.ImageOverlay(
            image=str(rgb_overlay_png),
            bounds=rgb_bounds,
            opacity=rgb_opacity,
            name=f"RGB {year}"
        ).add_to(m)
    else:
        st.info(
            f"Layer RGB untuk tahun {year} tidak ditemukan pada folder "
            f"`{RGB_LAYER_DIR}`."
        )

    # =====================================================
    # LAYER KEDALAMAN DI ATAS RGB
    # =====================================================
    folium.raster_layers.ImageOverlay(
        image=str(overlay_png),
        bounds=bounds,
        opacity=depth_opacity,
        name=f"Kedalaman Air {depth_choice} {year}"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # =====================================================
    # TAMBAHKAN MARKER TITIK YANG DIKLIK
    # =====================================================
    clicked_depth_value = None

    if clicked_point is not None:
        clicked_lat = clicked_point["lat"]
        clicked_lon = clicked_point["lng"]

        clicked_depth_value = sample_depth_value(
            tif_path=tif_path,
            lat=clicked_lat,
            lon=clicked_lon
        )

        if clicked_depth_value is not None:
            marker_popup = (
                f"<b>Titik Klik</b><br>"
                f"Latitude: {clicked_lat:.6f}<br>"
                f"Longitude: {clicked_lon:.6f}<br>"
                f"Kedalaman: {clicked_depth_value:.2f} m<br>"
            )
        else:
            marker_popup = (
                f"<b>Titik Klik</b><br>"
                f"Latitude: {clicked_lat:.6f}<br>"
                f"Longitude: {clicked_lon:.6f}<br>"
                f"Tidak ada nilai kedalaman valid."
            )

        folium.Marker(
            location=[clicked_lat, clicked_lon],
            popup=marker_popup,
            tooltip="Titik yang diklik",
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # =====================================================
    # LAYOUT PETA KIRI DAN INFORMASI KANAN
    # =====================================================
    st.subheader("Peta Interaktif Kedalaman Air")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        map_result = st_folium(
            m,
            width=None,
            height=700,
            key=f"depth_map_{depth_choice}_{year}",
            returned_objects=["last_clicked", "zoom", "center"]
        )
        
    # =====================================================
    # UPDATE INFORMASI ZOOM DAN CENTER PETA
    # =====================================================
    if "depth_map_zoom" not in st.session_state:
        st.session_state["depth_map_zoom"] = 9.2
    
    if "depth_map_center" not in st.session_state:
        st.session_state["depth_map_center"] = {
            "lat": center_lat,
            "lng": center_lon
        }
    
    if map_result is not None:
        if map_result.get("zoom") is not None:
            st.session_state["depth_map_zoom"] = map_result["zoom"]
    
        if map_result.get("center") is not None:
            st.session_state["depth_map_center"] = map_result["center"]

    # =====================================================
    # UPDATE TITIK KLIK
    # =====================================================
    last_clicked = map_result.get("last_clicked")

    if last_clicked is not None:
        old_clicked = st.session_state.get("depth_clicked_point")

        is_new_click = (
            old_clicked is None
            or abs(old_clicked["lat"] - last_clicked["lat"]) > 1e-10
            or abs(old_clicked["lng"] - last_clicked["lng"]) > 1e-10
        )

        if is_new_click:
            st.session_state["depth_clicked_point"] = last_clicked
            st.session_state["depth_clicked_key"] = current_depth_key
            st.rerun()

    # =====================================================
    # PANEL INFORMASI DI SAMPING KANAN
    # =====================================================
    with info_col:
        st.markdown("### Skala Peta")
    
        current_zoom = st.session_state.get("depth_map_zoom", 9.2)
        current_center = st.session_state.get(
            "depth_map_center",
            {
                "lat": center_lat,
                "lng": center_lon
            }
        )
    
        scale_lat = current_center["lat"]
    
        scale_denominator = estimate_scale_denominator(
            zoom=current_zoom,
            latitude=scale_lat
        )
        
        cm_scale, km_scale = format_scale_denominator(scale_denominator)
        scale_text = "1:" + cm_scale
    
        st.metric("Zoom Peta", current_zoom)
        st.metric("Perkiraan Skala", scale_text)
        
        st.caption(
            f"Keterangan: 1 cm pada layar ≈ {km_scale} km di dunia nyata"
        )
    
        st.markdown("---")
        st.markdown("### Informasi Titik")

        clicked_point = st.session_state.get("depth_clicked_point")

        if clicked_point is None:
            st.info("Klik salah satu titik pada peta.")
        else:
            lat = clicked_point["lat"]
            lon = clicked_point["lng"]

            depth_value = sample_depth_value(
                tif_path=tif_path,
                lat=lat,
                lon=lon
            )

            st.metric("Latitude", f"{lat:.6f}")
            st.metric("Longitude", f"{lon:.6f}")

            if depth_value is None:
                st.warning("Titik tidak memiliki nilai kedalaman valid.")
            else:
                st.metric("Kedalaman", f"{depth_value:.2f} m")
  
# =====================================================
# SKALA INFORMASI
# =====================================================
def estimate_scale_denominator(zoom, latitude, dpi=96):
    """
    Menghitung estimasi skala peta web dalam bentuk 1:N.
    Skala ini berubah sesuai zoom dan posisi lintang peta.
    """
    meters_per_pixel = (
        156543.03392
        * math.cos(math.radians(latitude))
        / (2 ** zoom)
    )

    denominator = meters_per_pixel * dpi / 0.0254

    return denominator


def format_scale_denominator(denominator):
    """
    Mengubah angka skala menjadi format cm dan km.

    Contoh:
    12500 ->
    cm = 12.500
    km = 0,125
    """

    if denominator is None or denominator <= 0:
        return "-", "-"

    rounded = int(round(denominator / 100) * 100)

    # format cm
    cm = f"{rounded:,}".replace(",", ".")

    # konversi ke km
    km_value = rounded / 100000

    # format km
    km = f"{km_value:.3f}".replace(".", ",")

    return cm, km

# =====================================================
# SIDEBAR NAVIGASI
# =====================================================
st.sidebar.title("Menu Visualisasi")

with st.sidebar:
    st.markdown("### Menu")

    page = option_menu(
        menu_title=None,
        options=[
            "Kedalaman Air",
            "Rumput Laut",
            "Terumbu Karang"
        ],
        icons=[
            "water",
            "flower1",
            "tree",
            "intersect"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0px",
                "background-color": "transparent"
            },
            "icon": {
                "color": "#4f6ef7",
                "font-size": "16px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0px",
                "padding": "10px 12px",
                "border-radius": "8px",
                "--hover-color": "#edf2ff"
            },
            "nav-link-selected": {
                "background-color": "#4f6ef7",
                "color": "white",
                "font-weight": "600"
            }
        }
    )


# =====================================================
# ROUTING HALAMAN
# =====================================================
if page == "Kedalaman Air":
    render_depth_page()

elif page == "Rumput Laut":
    render_two_maps_page(
        title="Visualisasi Zona Potensial Rumput Laut",
        folder=SEAWEED_DIR
    )

elif page == "Terumbu Karang":
    render_two_maps_page(
        title="Visualisasi Zona Potensial Terumbu Karang",
        folder=REEF_DIR
    )