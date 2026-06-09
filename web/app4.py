# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:40:11 2026

@author: Tiffany
"""

import json
import math
import os
import tempfile
from pathlib import Path
import base64

import branca.colormap as cm_branca
import folium
from matplotlib import colormaps
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from rasterio.vrt import WarpedVRT
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu


# ============================================================
# KONFIGURASI DASAR
# ============================================================
st.set_page_config(
    page_title="Web Pemetaan Potensial",
    layout="wide"
)

# Gunakan raw string r"..." agar path Windows aman dibaca Python.
RGB_DIR = Path(r"D:\TA\code\web\rgb_layer")
DEPTH_DIR = Path(r"D:\TA\code\depth")
SEAWEED_DIR = Path(r"D:\TA\code\web\geojsonSeaweed")
REEF_DIR = Path(r"D:\TA\code\web\geojsonReef")

# Tahun RGB yang digunakan pada halaman rumput laut dan terumbu karang.
# Supaya sederhana, halaman zona tidak diberi pilihan tahun RGB dulu.
DEFAULT_RGB_YEAR_FOR_ZONES = 2025

# Posisi awal jika file belum terbaca.
DEFAULT_CENTER = [-5.856545763743512, 137.87603646138956]
DEFAULT_ZOOM = 9

# Field popup dari GeoJSON.
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
    "Luas Zona",
    "Rata-rata Salinitas",
    "Rata-rata Kedalaman",
    "Rata-rata Suhu",
    "Rata-rata Sedimentasi"
]

# Field baru yang akan dibuat untuk menambahkan satuan pada popup.
POPUP_FORMATTED_FIELDS = [
    "popup_zone_id",
    "popup_area_ha",
    "popup_salinity_mean",
    "popup_depth_mean",
    "popup_temperature_mean",
    "popup_sedimentation_mean"
]


# ============================================================
# FUNGSI BANTU FILE
# ============================================================
def find_raster_file(folder, filename):
    """
    Mencari file raster berdasarkan nama file yang sudah pasti.
    """
    file_path = folder / filename

    if file_path.exists():
        return file_path

    return None


def find_geojson_file(folder, filename):
    """
    Mencari file GeoJSON berdasarkan nama file yang sudah pasti.
    """
    file_path = folder / filename

    if file_path.exists():
        return file_path

    return None


# ============================================================
# FUNGSI BANTU SKALA PETA
# ============================================================
def meters_per_pixel(latitude, zoom):
    """
    Menghitung estimasi meter per pixel berdasarkan zoom Web Mercator.

    Rumus ini bersifat perkiraan, bukan skala kartografi cetak.
    Nilai berubah ketika zoom berubah.
    """
    return 156543.03392 * math.cos(math.radians(latitude)) / (2 ** zoom)


def show_scale_info(view_state):
    """
    Menampilkan keterangan skala di kanan atas sebelum peta.
    """
    center = view_state.get("center", DEFAULT_CENTER)
    zoom = view_state.get("zoom", DEFAULT_ZOOM)

    lat = center[0]
    mpp = meters_per_pixel(lat, zoom)

    st.markdown(
        f"""
        <div style="text-align:right; font-size:14px; margin-bottom:8px;">
            <b>Skala tampilan:</b> Zoom {zoom} &nbsp;|&nbsp; ± {mpp:,.1f} m/pixel
        </div>
        """,
        unsafe_allow_html=True
    )


def update_view_state(view_key, map_data):
    """
    Menyimpan center dan zoom terakhir dari st_folium ke session_state.
    Zoom paling kecil dibatasi pada DEFAULT_ZOOM.
    """
    if not map_data:
        return

    center = map_data.get("center")
    zoom = map_data.get("zoom")

    if center is None or zoom is None:
        return

    if isinstance(center, dict):
        new_center = [center["lat"], center["lng"]]
    else:
        new_center = center

    # Pastikan zoom tidak lebih kecil dari 9
    new_zoom = max(int(zoom), DEFAULT_ZOOM)

    st.session_state[view_key] = {
        "center": new_center,
        "zoom": new_zoom
    }


def ensure_view_state(view_key, default_center=DEFAULT_CENTER, default_zoom=DEFAULT_ZOOM):
    """
    Membuat state awal peta jika belum ada.
    """
    if view_key not in st.session_state:
        st.session_state[view_key] = {
            "center": default_center,
            "zoom": default_zoom
        }

    return st.session_state[view_key]

def make_depth_colors(number_of_bins):
    """
    Membuat warna untuk raster kedalaman.
    Menggunakan colormap Blues dengan cara baru agar tidak muncul warning.
    """

    # Ambil colormap Blues, lalu ubah menjadi jumlah kelas yang dibutuhkan
    cmap = colormaps["Blues"].resampled(number_of_bins)

    colors = []

    for i in range(number_of_bins):
        rgba = cmap(i)
        colors.append(mcolors.to_hex(rgba))

    return colors

def make_black_pixels_transparent_from_raw(data, base_mask, threshold=0):
    """
    Membuat piksel hitam dari data raster asli menjadi transparan.

    Parameter:
    data      : data raster RGB sebelum stretch, bentuknya (3, height, width)
    base_mask : mask awal dari NoData
    threshold : batas nilai asli yang dianggap hitam

    Return:
    alpha : array transparansi
            0   = transparan
            255 = terlihat
    """

    # Ambil nilai asli masing-masing band
    raw_r = data[0].filled(0)
    raw_g = data[1].filled(0)
    raw_b = data[2].filled(0)

    # Piksel dianggap hitam jika nilai asli R, G, dan B kecil
    black_mask = (
        (raw_r <= threshold)
        & (raw_g <= threshold)
        & (raw_b <= threshold)
    )

    # Gabungkan mask NoData dan mask hitam
    final_mask = base_mask | black_mask

    # Area mask dibuat transparan
    alpha = np.where(final_mask, 0, 255).astype(np.uint8)

    return alpha


@st.cache_data(show_spinner=False)
def render_geotiff_to_png(
    tif_path_str,
    layer_type,
    max_depth=None,
    depth_bin_size=None,
    file_mtime=None
):
    """
    Mengubah GeoTIFF menjadi PNG sementara agar bisa dipasang ke Folium ImageOverlay.

    layer_type:
    - "rgb"   : untuk RGB GeoTIFF
    - "depth" : untuk raster kedalaman

    file_mtime dipakai agar cache diperbarui jika file berubah.
    """
    tif_path = Path(tif_path_str)

    cache_dir = Path(tempfile.gettempdir()) / "streamlit_geotiff_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
    f"{tif_path.stem}_{layer_type}_{max_depth}_{depth_bin_size}_{int(file_mtime)}.png"
    )
    png_path = cache_dir / cache_name

    if png_path.exists():
        # Bounds tetap perlu dibaca agar posisi overlay benar.
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                bounds = [
                    [vrt.bounds.bottom, vrt.bounds.left],
                    [vrt.bounds.top, vrt.bounds.right]
                ]

        return str(png_path), bounds

    with rasterio.open(tif_path) as src:
        # WarpedVRT digunakan agar raster dibaca dalam koordinat EPSG:4326.
        # Folium membutuhkan koordinat latitude-longitude.
        with WarpedVRT(src, crs="EPSG:4326") as vrt:
            bounds = [
                [vrt.bounds.bottom, vrt.bounds.left],
                [vrt.bounds.top, vrt.bounds.right]
            ]

            # Supaya aplikasi tidak terlalu berat, raster diturunkan resolusinya
            # jika ukuran file sangat besar.
            max_size = 1600
            scale = min(1, max_size / max(vrt.width, vrt.height))
            out_width = max(1, int(vrt.width * scale))
            out_height = max(1, int(vrt.height * scale))

            if layer_type == "rgb":
                if vrt.count >= 3:
                    data = vrt.read(
                        [1, 2, 3],
                        out_shape=(3, out_height, out_width),
                        masked=True
                    )
            
                    # Mask bawaan dari raster, misalnya NoData
                    mask = (
                        np.ma.getmaskarray(data[0])
                        | np.ma.getmaskarray(data[1])
                        | np.ma.getmaskarray(data[2])
                    )
            
                    # Membuat alpha dari data asli, bukan dari RGB hasil stretch
                    alpha = make_black_pixels_transparent_from_raw(
                        data=data,
                        base_mask=mask,
                        threshold=0
                    )
            
                    # Setelah alpha dibuat, baru lakukan stretch RGB
                    r = data[0]
                    g = data[1]
                    b = data[2]
            
                    rgb = np.dstack([r, g, b])
            
                else:
                    data = vrt.read(
                        1,
                        out_shape=(out_height, out_width),
                        masked=True
                    )
            
                    gray = data
                    rgb = np.dstack([gray, gray, gray])
            
                    mask = np.ma.getmaskarray(data)
            
                    black_mask = gray <= 0
                    final_mask = mask | black_mask
            
                    alpha = np.where(final_mask, 0, 255).astype(np.uint8)
            
                rgba = np.dstack([rgb, alpha])

            elif layer_type == "depth":
                data = vrt.read(
                    1,
                    out_shape=(out_height, out_width),
                    masked=True
                )
            
                # Ubah data menjadi float agar bisa menyimpan NaN
                values = data.filled(np.nan).astype(float)
            
                # Mask awal dari raster
                mask = np.ma.getmaskarray(data) | np.isnan(values)
            
                # Jika ada nilai NoData dari metadata raster, ikut dimask
                if vrt.nodata is not None:
                    mask = mask | (values == vrt.nodata)
            
                # Ubah nilai kedalaman menjadi positif
                # Contoh: -10 m menjadi 10 m
                values = np.abs(values)
            
                # =====================================================
                # BAGIAN PENTING UNTUK MENGHILANGKAN BORDER UNGU
                # =====================================================
                # Border biasanya terbaca sebagai 0.
                # Jika 0 tidak dimask, maka 0 akan masuk bin pertama
                # dan muncul sebagai warna ungu pada colormap viridis.
                min_valid_depth = 0.01
            
                mask = (
                    mask
                    | (values <= min_valid_depth)
                    | (values > max_depth)
                )
            
                # Membuat interval kedalaman
                bins = np.arange(0, max_depth + depth_bin_size, depth_bin_size)
                number_of_bins = len(bins) - 1
                colors = make_depth_colors(number_of_bins)
            
                # Siapkan RGBA.
                # Nilai awal semuanya transparan.
                rgba = np.zeros((out_height, out_width, 4), dtype=np.uint8)
            
                # Klasifikasi nilai kedalaman ke interval warna
                bin_index = np.digitize(values, bins, right=False) - 1
                bin_index = np.clip(bin_index, 0, number_of_bins - 1)
            
                for i, color_hex in enumerate(colors):
                    rgb_color = mcolors.to_rgb(color_hex)
                    rgb_color = [int(c * 255) for c in rgb_color]
            
                    # Hanya area valid yang diberi warna
                    area = (bin_index == i) & (~mask)
            
                    rgba[area, 0] = rgb_color[0]
                    rgba[area, 1] = rgb_color[1]
                    rgba[area, 2] = rgb_color[2]
                    rgba[area, 3] = 210

            else:
                raise ValueError("layer_type harus berupa 'rgb' atau 'depth'.")

    Image.fromarray(rgba).save(png_path)

    return str(png_path), bounds


def add_geotiff_overlay(
    folium_map,
    tif_path,
    layer_name,
    layer_type,
    opacity=0.8,
    max_depth=None,
    depth_bin_size=None
):
    """
    Menambahkan GeoTIFF sebagai ImageOverlay pada Folium.
    """
    if tif_path is None:
        return None

    file_mtime = os.path.getmtime(tif_path)

    png_path, bounds = render_geotiff_to_png(
        tif_path_str=str(tif_path),
        layer_type=layer_type,
        max_depth=max_depth,
        depth_bin_size=depth_bin_size,
        file_mtime=file_mtime
    )

    folium.raster_layers.ImageOverlay(
        image=png_path,
        bounds=bounds,
        name=layer_name,
        opacity=opacity,
        interactive=False,
        cross_origin=False,
        zindex=1
    ).add_to(folium_map)

    return bounds


def add_depth_legend(folium_map, max_depth, depth_bin_size):
    """
    Menambahkan legenda kedalaman.

    Untuk 10 m dipakai interval 1 m.
    Untuk 35 m lebih disarankan interval 5 m agar legenda tidak terlalu panjang.
    """
    bins = np.arange(0, max_depth + depth_bin_size, depth_bin_size)
    colors = make_depth_colors(len(bins) - 1)

    colormap = cm_branca.StepColormap(
        colors=colors,
        index=bins,
        vmin=0,
        vmax=max_depth,
        caption=f"Kedalaman air (m), interval {depth_bin_size} m"
    )

    colormap.add_to(folium_map)


# ============================================================
# FUNGSI BANTU GEOJSON
# ============================================================
def format_number(value, unit="", decimals=2):
    """
    Memformat angka dan menambahkan satuan.
    """
    try:
        if value is None:
            return "-"

        number = float(value)

        if math.isnan(number):
            return "-"
        
        # Format angka dengan 2 angka di belakang koma
        formatted_number = f"{number:.{decimals}f}"

        # Ubah pemisah desimal dari titik menjadi koma
        formatted_number = formatted_number.replace(".", ",")

        return f"{formatted_number} {unit}".strip()

    except Exception:
        return "-"


def load_geojson_with_popup(geojson_path):
    """
    Membaca GeoJSON dan menambahkan field khusus untuk popup.

    Field asli:
    - area_ha
    - salinity_mean
    - depth_mean
    - temperature_mean
    - sedimentation_mean

    Field baru:
    - popup_area_ha, dst.
    """
    if geojson_path is None:
        return None, 0

    with open(geojson_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    features = data.get("features", [])

    zone_ids = set()

    for feature in features:
        properties = feature.setdefault("properties", {})

        zone_id = properties.get("zone_id", "-")

        if zone_id != "-":
            zone_ids.add(str(zone_id))

        properties["popup_zone_id"] = str(zone_id)
        properties["popup_area_ha"] = format_number(properties.get("area_ha"), "ha")
        properties["popup_salinity_mean"] = format_number(properties.get("salinity_mean"), "psu")
        properties["popup_depth_mean"] = format_number(properties.get("depth_mean"), "m")
        properties["popup_temperature_mean"] = format_number(properties.get("temperature_mean"), "°C")
        properties["popup_sedimentation_mean"] = format_number(properties.get("sedimentation_mean"), "g/m³")

    # Jika zone_id tersedia, hitung jumlah zona unik.
    # Jika tidak ada, hitung jumlah feature.
    zone_count = len(zone_ids) if len(zone_ids) > 0 else len(features)

    return data, zone_count


def get_geojson_bounds(geojson_data):
    """
    Mengambil bounds GeoJSON secara sederhana.

    GeoJSON memakai urutan koordinat [longitude, latitude].
    Folium memakai [latitude, longitude].
    """
    if geojson_data is None:
        return None

    points = []

    def walk_coordinates(coords):
        if not isinstance(coords, list):
            return

        # Jika coords berbentuk [lon, lat]
        if (
            len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
        ):
            lon = coords[0]
            lat = coords[1]

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append([lat, lon])

        else:
            for item in coords:
                walk_coordinates(item)

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry")

        if geometry is None:
            continue

        walk_coordinates(geometry.get("coordinates", []))

    if not points:
        return None

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    return [
        [min(lats), min(lons)],
        [max(lats), max(lons)]
    ]


def bounds_to_center(bounds):
    """
    Mengubah bounds menjadi titik tengah peta.
    """
    if bounds is None:
        return DEFAULT_CENTER

    south, west = bounds[0]
    north, east = bounds[1]

    return [
        (south + north) / 2,
        (west + east) / 2
    ]


def merge_bounds(bounds_list):
    """
    Menggabungkan beberapa bounds menjadi satu bounds besar.
    """
    valid_bounds = [b for b in bounds_list if b is not None]

    if not valid_bounds:
        return None

    south = min(b[0][0] for b in valid_bounds)
    west = min(b[0][1] for b in valid_bounds)
    north = max(b[1][0] for b in valid_bounds)
    east = max(b[1][1] for b in valid_bounds)

    return [[south, west], [north, east]]


def add_geojson_layer(folium_map, geojson_data, layer_name, fill_color):
    """
    Menambahkan GeoJSON ke peta Folium.
    """
    if geojson_data is None:
        return

    folium.GeoJson(
        data=geojson_data,
        name=layer_name,
        style_function=lambda feature: {
            "fillColor": fill_color,
            "color": fill_color,
            "weight": 1,
            "fillOpacity": 0.55,
        },
        highlight_function=lambda feature: {
            "weight": 3,
            "fillOpacity": 0.75,
        },
        popup=folium.GeoJsonPopup(
            fields=POPUP_FORMATTED_FIELDS,
            aliases=POPUP_ALIASES,
            labels=True,
            localize=True,
            max_width=350
        )
    ).add_to(folium_map)
    
def summarize_geojson(geojson_data):
    """
    Membuat ringkasan informasi dari GeoJSON zona potensial.

    Informasi yang dihitung:
    - jumlah zona
    - total luas zona
    - rata-rata luas zona
    - zona terluas
    - rata-rata salinitas
    - rata-rata kedalaman
    - rata-rata suhu
    - rata-rata sedimentasi
    """

    if geojson_data is None:
        return {
            "zone_count": 0,
            "total_area": 0,
            "mean_area": 0,
            "largest_zone_id": "-",
            "largest_area": 0,
            "mean_salinity": None,
            "mean_depth": None,
            "mean_temperature": None,
            "mean_sedimentation": None,
            "table_data": []
        }

    features = geojson_data.get("features", [])

    table_data = []

    for feature in features:
        prop = feature.get("properties", {})

        row = {
            "ID Zona": prop.get("zone_id"),
            "Luas Zona (ha)": prop.get("area_ha"),
            "Rata-rata Salinitas (psu)": prop.get("salinity_mean"),
            "Rata-rata Kedalaman (m)": prop.get("depth_mean"),
            "Rata-rata Suhu (°C)": prop.get("temperature_mean"),
            "Rata-rata Sedimentasi (g/m³)": prop.get("sedimentation_mean")
        }

        table_data.append(row)

    if len(table_data) == 0:
        return {
            "zone_count": 0,
            "total_area": 0,
            "mean_area": 0,
            "largest_zone_id": "-",
            "largest_area": 0,
            "mean_salinity": None,
            "mean_depth": None,
            "mean_temperature": None,
            "mean_sedimentation": None,
            "table_data": []
        }

    # Ambil nilai numerik yang valid
    areas = [
        float(row["Luas Zona (ha)"])
        for row in table_data
        if row["Luas Zona (ha)"] is not None
    ]

    salinities = [
        float(row["Rata-rata Salinitas (psu)"])
        for row in table_data
        if row["Rata-rata Salinitas (psu)"] is not None
    ]

    depths = [
        abs(float(row["Rata-rata Kedalaman (m)"]))
        for row in table_data
        if row["Rata-rata Kedalaman (m)"] is not None
    ]

    temperatures = [
        float(row["Rata-rata Suhu (°C)"])
        for row in table_data
        if row["Rata-rata Suhu (°C)"] is not None
    ]

    sedimentations = [
        float(row["Rata-rata Sedimentasi (g/m³)"])
        for row in table_data
        if row["Rata-rata Sedimentasi (g/m³)"] is not None
    ]

    total_area = sum(areas) if areas else 0
    mean_area = total_area / len(areas) if areas else 0

    # Cari zona terluas
    largest_row = None

    for row in table_data:
        area = row["Luas Zona (ha)"]

        if area is None:
            continue

        if largest_row is None or float(area) > float(largest_row["Luas Zona (ha)"]):
            largest_row = row

    if largest_row is not None:
        largest_zone_id = largest_row["ID Zona"]
        largest_area = float(largest_row["Luas Zona (ha)"])
    else:
        largest_zone_id = "-"
        largest_area = 0

    return {
        "zone_count": len(table_data),
        "total_area": total_area,
        "mean_area": mean_area,
        "largest_zone_id": largest_zone_id,
        "largest_area": largest_area,
        "mean_salinity": np.mean(salinities) if salinities else None,
        "mean_depth": np.mean(depths) if depths else None,
        "mean_temperature": np.mean(temperatures) if temperatures else None,
        "mean_sedimentation": np.mean(sedimentations) if sedimentations else None,
        "table_data": table_data
    }

def show_small_summary_card(title, value):
    """
    Menampilkan kartu ringkasan kecil dengan ukuran tulisan yang bisa diatur.
    """

    st.markdown(
        f"""
        <div style="
            padding: 12px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        ">
            <div style="
                font-weight: 700;
                font-size: 16px;
                color: #0b5394;
                margin-bottom: 4px;
            ">
                {title}
            </div>
            <div style="
                font-size: 16px;
                font-weight: 500;
                color: #555;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_zone_summary(summary):
    """
    Menampilkan ringkasan zona dengan ukuran tulisan lebih kecil.
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_small_summary_card(
            "Jumlah Zona",
            f"{summary['zone_count']} zona"
        )

    with col2:
        show_small_summary_card(
            "Total Luas",
            format_number(summary["total_area"], "ha")
        )

    with col3:
        show_small_summary_card(
            "Rata-rata Luas",
            format_number(summary["mean_area"], "ha")
        )

    with col4:
        # Sengaja dikosongkan agar struktur 4 kolom tetap sama
        st.markdown("&nbsp;", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        show_small_summary_card(
            "Rata-rata Salinitas",
            format_number(summary["mean_salinity"], "psu")
        )

    with col6:
        show_small_summary_card(
            "Rata-rata Kedalaman",
            format_number(summary["mean_depth"], "m")
        )

    with col7:
        show_small_summary_card(
            "Rata-rata Suhu",
            format_number(summary["mean_temperature"], "°C")
        )

    with col8:
        show_small_summary_card(
            "Rata-rata Sedimentasi",
            format_number(summary["mean_sedimentation"], "g/m³")
        )
        
def show_zone_table(summary):
    """
    Menampilkan tabel detail zona.
    """

    if len(summary["table_data"]) == 0:
        st.info("Belum ada data zona yang dapat ditampilkan.")
        return

    st.dataframe(
        summary["table_data"],
        width="stretch",
        hide_index=True
    )


# ============================================================
# FUNGSI PEMBUAT PETA
# ============================================================
def create_base_map(center, zoom):
    """
    Membuat peta dasar Folium.

    min_zoom dibuat tetap 9 agar pengguna masih bisa zoom out
    sampai zoom 9, meskipun sebelumnya peta sedang berada pada zoom 10,
    11, atau lebih besar.
    """

    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        min_zoom=DEFAULT_ZOOM,   # batas zoom out paling kecil
        max_zoom=18,
        control_scale=True,
        max_bounds=False
    )

# ============================================================
# FUNGSI PENAMBAH GAMBAR
# ============================================================
INSET_IMAGE_PATH = Path(r"D:\TA\code\web\layer.png")

def add_inset_image(folium_map, image_path, position="bottomleft", width=260):
    """
    Menambahkan gambar inset di atas peta Folium.

    position:
    - "bottomleft"
    - "bottomright"
    - "topleft"
    - "topright"
    """

    image_path = Path(image_path)

    # Jika file tidak ditemukan, fungsi berhenti
    if not image_path.exists():
        return

    # Baca gambar menjadi base64
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    image_data = f"data:image/png;base64,{encoded}"

    # Atur posisi gambar
    if position == "bottomleft":
        position_style = "left: 20px; bottom: 55px;"
    elif position == "bottomright":
        position_style = "right: 20px; bottom: 55px;"
    elif position == "topleft":
        position_style = "left: 20px; top: 20px;"
    elif position == "topright":
        position_style = "right: 20px; top: 90px;"
    else:
        position_style = "left: 20px; bottom: 55px;"

    html = f"""
    <div style="
        position: fixed;
        {position_style}
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.85);
        border: 2px solid black;
        padding: 4px;
    ">
        <img src="{image_data}" style="
            width: {width}px;
            display: block;
        ">
    </div>
    """

    folium_map.get_root().html.add_child(folium.Element(html))
    

# ============================================================
# HALAMAN 1: PEMETAAN KEDALAMAN AIR
# ============================================================
def page_depth_mapping():
    st.title("Pemetaan Kedalaman Air")
    st.header("Perairan Pesisir Kabupaten Asmat, Papua Selatan")

    col_filter_1, col_filter_2 = st.columns(2)

    with col_filter_1:
        depth_label = st.selectbox(
            "Pilih kedalaman maksimum",
            options=["10 m", "35 m"],
            index=0
        )

    with col_filter_2:
        year = st.selectbox(
            "Pilih tahun",
            options=[2023, 2024, 2025],
            index=2
        )

    # Mengubah label menjadi format yang dipakai nama file.
    depth_code = depth_label.replace(" ", "")  # "10 m" menjadi "10m"
    max_depth = 10 if depth_code == "10m" else 35

    # Untuk 10 m, interval 1 m masih mudah dibaca.
    # Untuk 35 m, interval 5 m lebih sederhana agar legenda tidak terlalu panjang.
    depth_bin_size = 1 if max_depth == 10 else 5

    rgb_file = find_raster_file(RGB_DIR, f"rgb_{year}.tif")
    depth_file = find_raster_file(DEPTH_DIR, f"depth_{depth_code}_{year}.tif")
    
    if rgb_file is None:
        st.warning(f"File RGB untuk tahun {year} tidak ditemukan di {RGB_DIR}")

    if depth_file is None:
        st.warning(f"File kedalaman {depth_code} tahun {year} tidak ditemukan di {DEPTH_DIR}")

    view_key = "view_depth"
    
    st.markdown(
        f"""
        <div style="text-align:left; font-size:20px; margin-bottom:8px;">
            <b>Hasil Pemetaan Kedalaman Air Maksimal {depth_label} Tahun {year}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("Keterangan metode pemetaan"):
        st.write(
            """
            Halaman menampilkan hasil pemetaan kedalaman air wilayah perairan pesisir
            Kabupaten Asmat dengan model ***Random Forest Regression***. 
            """
        )
        
    # Keterangan skala diletakkan di kanan atas sebelum peta.
    view_state = ensure_view_state(view_key)
    show_scale_info(view_state)
    
    st.caption(
        "Catatan: Jika salah satu peta digeser atau di-zoom, peta lainnya akan mengikuti setelah Streamlit melakukan pembaruan tampilan."
    )

    m = create_base_map(
        center=view_state["center"],
        zoom=9
    )

    # Layer RGB.
    add_geotiff_overlay(
        folium_map=m,
        tif_path=rgb_file,
        layer_name=f"RGB {year}",
        layer_type="rgb",
        opacity=0.90
    )
    
    # Layer kedalaman.
    add_geotiff_overlay(
        folium_map=m,
        tif_path=depth_file,
        layer_name=f"Kedalaman {depth_label} - {year}",
        layer_type="depth",
        opacity=1,
        max_depth=max_depth,
        depth_bin_size=depth_bin_size
    )
    
    # Tambahkan inset lokasi
    add_inset_image(
        folium_map=m,
        image_path=INSET_IMAGE_PATH,
        position="bottomleft",
        width=260
    )
    
    if depth_file is not None:
        add_depth_legend(m, max_depth=max_depth, depth_bin_size=depth_bin_size)
    
    folium.LayerControl(collapsed=False).add_to(m)

    map_data = st_folium(
        m,
        height=620,
        width="stretch",
        returned_objects=["center", "zoom"],
        key="map_depth"
    )

    update_view_state(view_key, map_data)


# ============================================================
# HALAMAN 2 DAN 3: AREA POTENSIAL
# ============================================================

def page_potential_area(page_title, geojson_dir, view_key, color_no_check, color_check):
    st.title(page_title)
    st.header("Perairan Pesisir Kabupaten Asmat, Papua Selatan")

    # Tentukan pilihan kedalaman berdasarkan halaman
    if view_key == "view_seaweed":
        depth_options = ["10 m"]
    else:
        depth_options = ["10 m", "35 m"]

    depth_label = st.selectbox(
        "Pilih kedalaman",
        options=depth_options,
        index=0,
        key=f"{view_key}_depth_selectbox"
    )

    depth_code = depth_label.replace(" ", "")

    no_check_file = find_geojson_file(
        geojson_dir,
        f"zona_potensial_{depth_code}_noCheck4factor.geojson"
    )
    
    check_file = find_geojson_file(
        geojson_dir,
        f"zona_potensial_{depth_code}_check4factor.geojson"
    )

    rgb_file = find_raster_file(RGB_DIR, f"rgb_{DEFAULT_RGB_YEAR_FOR_ZONES}.tif")
    
    st.markdown(
        f"""
        <div style="text-align:left; font-size:20px; margin-bottom:8px;">
            <b>Hasil Pemetaan Area Potensial Kedalaman Maksimal {depth_code} Tahun 2025</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if no_check_file is None and check_file is None:
        st.warning(
            f"Tidak tersedia zona potensial untuk kedalaman maksimal {depth_code}."
        )
        return
    
    if rgb_file is None:
        st.warning(
            f"File RGB tahun {DEFAULT_RGB_YEAR_FOR_ZONES} tidak ditemukan di {RGB_DIR}"
        )
        return

    no_check_geojson, no_check_count = load_geojson_with_popup(no_check_file)
    check_geojson, check_count = load_geojson_with_popup(check_file)
    
    no_check_summary = summarize_geojson(no_check_geojson)
    check_summary = summarize_geojson(check_geojson)
    
    with st.expander("Keterangan hasil pemetaan"):
        st.write(
            """
            Halaman ini menampilkan hasil zona area potensial dengan model ***Random Forest Classification*** 
            untuk hasil pemetaan dan penyaringan faktor lingkungan (salinitas, kedalaman, suhu, dan sedimentasi).
            - Peta <b>kiri</b> menunjukkan hasil zona <b>tanpa pengecekan ulang</b> 
            empat faktor lingkungan. 
            - Peta <b>kanan</b> menunjukkan hasil zona dengan <b>pengecekan ulang</b> 
            empat faktor lingkungan.
            """,
            unsafe_allow_html=True
        )
        
    # Keterangan skala diletakkan di kanan atas sebelum peta.
    view_state = ensure_view_state(
        view_key=view_key,
        default_center=DEFAULT_CENTER,
        default_zoom=DEFAULT_ZOOM
    )
    show_scale_info(view_state)
    
    st.caption(
        "Catatan: Jika salah satu peta digeser atau di-zoom, peta lainnya akan mengikuti setelah Streamlit melakukan pembaruan tampilan."
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(
            """
            <div style="text-align:left; font-size:18px;margin-bottom:8px">
                Zona Gabungan Area Potensial <u>Tanpa Pengecekan</u> Faktor Lingkungan
            </div>
            """,
            unsafe_allow_html=True
        )

        m_left = create_base_map(
            center=st.session_state[view_key]["center"],
            zoom=st.session_state[view_key]["zoom"]
        )

        add_geotiff_overlay(
            folium_map=m_left,
            tif_path=rgb_file,
            layer_name=f"RGB {DEFAULT_RGB_YEAR_FOR_ZONES}",
            layer_type="rgb",
            opacity=0.85
        )

        add_geojson_layer(
            folium_map=m_left,
            geojson_data=no_check_geojson,
            layer_name="Zona noCheck4factor",
            fill_color="#2ca25f"
        )

        folium.LayerControl(collapsed=False).add_to(m_left)

        left_data = st_folium(
            m_left,
            height=560,
            width="stretch",
            returned_objects=["center", "zoom"],
            key=f"{view_key}_left_map"
        )

        update_view_state(view_key, left_data)

        st.info(f"Jumlah zona: {no_check_count} zona")
        show_zone_summary(no_check_summary)
        with st.expander("Lihat tabel detail zona"):
            show_zone_table(no_check_summary)

    with right_col:
        st.markdown(
            """
            <div style="text-align:left; font-size:18px;margin-bottom:8px">
                Zona Gabungan Area Potensial <u>Dengan Pengecekan</u> Faktor Lingkungan
            </div>
            """,
            unsafe_allow_html=True
        )

        m_right = create_base_map(
            center=st.session_state[view_key]["center"],
            zoom=st.session_state[view_key]["zoom"]
        )

        add_geotiff_overlay(
            folium_map=m_right,
            tif_path=rgb_file,
            layer_name=f"RGB {DEFAULT_RGB_YEAR_FOR_ZONES}",
            layer_type="rgb",
            opacity=0.85
        )

        add_geojson_layer(
            folium_map=m_right,
            geojson_data=check_geojson,
            layer_name="Zona check4factor",
            fill_color="#08519c"
        )

        folium.LayerControl(collapsed=False).add_to(m_right)

        right_data = st_folium(
            m_right,
            height=560,
            width="stretch",
            returned_objects=["center", "zoom"],
            key=f"{view_key}_right_map"
        )

        update_view_state(view_key, right_data)

        st.info(f"Jumlah zona: {check_count} zona")
        show_zone_summary(check_summary)
        with st.expander("Lihat tabel detail zona"):
            show_zone_table(check_summary)


# ============================================================
# MENU UTAMA
# ============================================================

with st.sidebar:
    selected_page = option_menu(
        menu_title="Menu",
        options=[
            "Kedalaman Air",
            "Rumput Laut",
            "Terumbu Karang"
        ],
        default_index=0
    )

if selected_page == "Kedalaman Air":
    page_depth_mapping()

elif selected_page == "Rumput Laut":
    page_potential_area(
        page_title="Area Potensial Rumput Laut",
        geojson_dir=SEAWEED_DIR,
        view_key="view_seaweed",
        color_no_check="#2ca25f",
        color_check="#08519c"
    )

elif selected_page == "Terumbu Karang":
    page_potential_area(
        page_title="Area Potensial Terumbu Karang",
        geojson_dir=REEF_DIR,
        view_key="view_reef",
        color_no_check="#2ca25f",
        color_check="#08519c"
    )