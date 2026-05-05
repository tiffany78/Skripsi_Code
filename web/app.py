#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:13:34 2026

@author: tipanoii
"""

from pathlib import Path

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Web Map Zona Potensial",
    layout="wide"
)

st.title("Web Visualisasi Zona Potensial")
st.write(
    "Pilih salah satu file GeoJSON, lalu klik area/zona pada peta "
    "untuk melihat informasi zona."
)


# =====================================================
# KONFIGURASI PATH
# =====================================================
# Karena app.py berada di /Users/tipanoii/doc/TA/code/web/app.py,
# maka BASE_DIR otomatis mengarah ke folder /web.
BASE_DIR = Path(__file__).resolve().parent

# Membaca semua file .geojson yang berada langsung di folder /web
GEOJSON_FILES = sorted((BASE_DIR / "geojson").glob("*.geojson"))


# =====================================================
# CEK FILE GEOJSON
# =====================================================
if not GEOJSON_FILES:
    st.error(
        "Tidak ada file GeoJSON ditemukan di folder:\n\n"
        f"{BASE_DIR}"
    )
    st.stop()


# =====================================================
# FUNGSI LOAD DATA
# =====================================================
@st.cache_data
def load_geojson(file_path: str):
    gdf = gpd.read_file(file_path)

    if gdf.empty:
        return gdf

    # Pastikan CRS cocok untuk web map
    if gdf.crs is None:
        # Jika GeoJSON sudah benar, biasanya EPSG:4326.
        # Baris ini hanya sebagai fallback.
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


# =====================================================
# SIDEBAR: PILIH FILE
# =====================================================
st.sidebar.header("Pengaturan Peta")

file_options = {
    file.name: file
    for file in GEOJSON_FILES
}

selected_file_name = st.sidebar.selectbox(
    "Pilih file GeoJSON",
    options=list(file_options.keys())
)

selected_file = file_options[selected_file_name]

st.sidebar.write("File aktif:")
st.sidebar.code(selected_file.name)

# =====================================================
# LOAD GEOJSON TERPILIH
# =====================================================
gdf = load_geojson(str(selected_file))

if gdf.empty:
    st.warning("GeoJSON terpilih tidak memiliki data zona.")
    st.stop()


# =====================================================
# KOLOM YANG AKAN DITAMPILKAN
# =====================================================
popup_fields = [
    "zone_id",
    "area_ha",
    "salinity_mean",
    "depth_mean",
    "temperature_mean",
    "sedimentation_mean"
]

popup_aliases = [
    "ID Zona",
    "Luas Zona (ha)",
    "Rata-rata Salinitas",
    "Rata-rata Kedalaman",
    "Rata-rata Suhu",
    "Rata-rata Sedimentasi"
]

# Pastikan semua kolom tersedia
missing_columns = [
    col for col in popup_fields
    if col not in gdf.columns
]

if missing_columns:
    st.error(
        "Ada kolom yang tidak ditemukan pada GeoJSON:\n\n"
        + ", ".join(missing_columns)
    )
    st.stop()


# =====================================================
# MEMBUAT PETA
# =====================================================
minx, miny, maxx, maxy = gdf.total_bounds

center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=8,
    tiles="OpenStreetMap"
)


# =====================================================
# STYLE POLYGON
# =====================================================
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


# =====================================================
# POPUP DAN TOOLTIP
# =====================================================
popup = folium.GeoJsonPopup(
    fields=popup_fields,
    aliases=popup_aliases,
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


# =====================================================
# TAMBAHKAN GEOJSON KE PETA
# =====================================================
folium.GeoJson(
    data=gdf,
    name="Zona Potensial",
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=tooltip,
    popup=popup
).add_to(m)

m.fit_bounds([
    [miny, minx],
    [maxy, maxx]
])

folium.LayerControl().add_to(m)


# =====================================================
# TAMPILKAN PETA
# =====================================================
st.subheader("Peta Zona Potensial")

st_folium(
    m,
    width=None,
    height=700
)


# =====================================================
# TAMPILKAN TABEL RINGKASAN
# =====================================================
st.subheader("Tabel Ringkasan Zona")

st.dataframe(
    gdf[popup_fields],
    use_container_width=True
)