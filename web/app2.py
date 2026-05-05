#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:12:59 2026

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
    page_title="Perbandingan Zona Potensial",
    layout="wide"
)

st.title("Web Visualisasi Zona Potensial Side by Side")
st.write(
    "Pilih dua file GeoJSON untuk dibandingkan secara berdampingan. "
    "Klik salah satu zona pada peta untuk melihat informasi ringkas zona."
)


# =====================================================
# KONFIGURASI PATH
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
GEOJSON_DIR = BASE_DIR / "geojson"
GEOJSON_FILES = sorted(GEOJSON_DIR.glob("*.geojson"))

if not GEOJSON_DIR.exists():
    st.error(f"Folder GeoJSON tidak ditemukan:\n\n{GEOJSON_DIR}")
    st.stop()

if not GEOJSON_FILES:
    st.error(f"Tidak ada file GeoJSON ditemukan di folder:\n\n{GEOJSON_DIR}")
    st.stop()


# =====================================================
# KOLOM ATRIBUT YANG DITAMPILKAN
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
# FUNGSI LOAD DATA
# =====================================================
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


# =====================================================
# FUNGSI VALIDASI KOLOM
# =====================================================
def validate_columns(gdf, file_name):
    missing_columns = [
        col for col in POPUP_FIELDS
        if col not in gdf.columns
    ]

    if missing_columns:
        st.error(
            f"File `{file_name}` tidak memiliki kolom berikut:\n\n"
            + ", ".join(missing_columns)
        )
        return False

    return True


# =====================================================
# FUNGSI MEMBUAT PETA
# =====================================================
def create_map(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds

    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap"
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

    m.fit_bounds([
        [miny, minx],
        [maxy, maxx]
    ])

    folium.LayerControl().add_to(m)

    return m


# =====================================================
# PILIHAN FILE
# =====================================================
file_options = {
    file.name: file
    for file in GEOJSON_FILES
}

file_names = list(file_options.keys())


# =====================================================
# LAYOUT SIDE BY SIDE
# =====================================================
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Peta Kiri")

    selected_left_name = st.selectbox(
        "Pilih GeoJSON untuk peta kiri",
        options=file_names,
        index=0,
        key="left_geojson"
    )

    selected_left_file = file_options[selected_left_name]
    gdf_left = load_geojson(str(selected_left_file))

    if gdf_left.empty:
        st.warning("GeoJSON kiri tidak memiliki data zona.")
    elif validate_columns(gdf_left, selected_left_name):
        left_map = create_map(gdf_left)

        st_folium(
            left_map,
            width=None,
            height=650,
            key="left_map"
        )

        with st.expander("Tabel ringkasan peta kiri"):
            st.dataframe(
                gdf_left[POPUP_FIELDS],
                use_container_width=True
            )


with right_col:
    st.subheader("Peta Kanan")

    default_right_index = 1 if len(file_names) > 1 else 0

    selected_right_name = st.selectbox(
        "Pilih GeoJSON untuk peta kanan",
        options=file_names,
        index=default_right_index,
        key="right_geojson"
    )

    selected_right_file = file_options[selected_right_name]
    gdf_right = load_geojson(str(selected_right_file))

    if gdf_right.empty:
        st.warning("GeoJSON kanan tidak memiliki data zona.")
    elif validate_columns(gdf_right, selected_right_name):
        right_map = create_map(gdf_right)

        st_folium(
            right_map,
            width=None,
            height=650,
            key="right_map"
        )

        with st.expander("Tabel ringkasan peta kanan"):
            st.dataframe(
                gdf_right[POPUP_FIELDS],
                use_container_width=True
            )