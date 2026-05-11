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
from streamlit_option_menu import option_menu


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
OVERLAP_DIR = BASE_DIR / "geojsonOverlap"
DEPTH_PNG_DIR = BASE_DIR / "depth_png"


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


def create_restricted_map(gdf, min_zoom=8):
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
        tiles="OpenStreetMap",

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
            left_map = create_restricted_map(gdf_left, min_zoom=8)
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
            right_map = create_restricted_map(gdf_right, min_zoom=8)
            st_folium(right_map, width=None, height=650, key=f"{title}_right_map")

            with st.expander("Tabel ringkasan peta kanan"):
                st.dataframe(
                    gdf_right[POPUP_FIELDS],
                    use_container_width=True
                )


def render_depth_page():
    st.header("Pemetaan Kedalaman Air")

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

    selected_folder = DEPTH_PNG_DIR / depth_choice

    if not selected_folder.exists():
        st.error(f"Folder PNG kedalaman tidak ditemukan:\n\n{selected_folder}")
        return

    png_candidates = sorted(selected_folder.glob(f"*{year}*.png"))

    if not png_candidates:
        st.warning(
            f"Tidak ada PNG kedalaman untuk tahun {year} "
            f"pada folder:\n\n{selected_folder}"
        )
        return

    selected_png = png_candidates[0]

    st.write(f"Menampilkan file: `{selected_png.name}`")

    st.image(
        str(selected_png),
        use_container_width=True
    )


def render_overlap_page():
    st.header("Gabungan Rumput Laut + Terumbu Karang")

    st.info(
        "Halaman ini disiapkan untuk menampilkan zona tumpang tindih antara "
        "hasil pemetaan rumput laut dan terumbu karang. Karena GeoTIFF/GeoJSON "
        "gabungan belum tersedia, peta ditampilkan kosong terlebih dahulu."
    )

    # Bounds sementara sekitar wilayah Indonesia timur.
    # Nanti dapat diganti dengan bounds AOI Asmat atau bounds hasil overlap.
    m = folium.Map(
        location=[-5.5, 138.5],
        zoom_start=7,
        tiles="OpenStreetMap",
        min_zoom=10,
        max_zoom=18,
        control_scale=True
    )

    st_folium(m, width=None, height=700, key="overlap_empty_map")


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
            "Terumbu Karang",
            "Gabungan"
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

elif page == "Gabungan":
    render_overlap_page()