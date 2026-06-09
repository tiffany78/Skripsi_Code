#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:09:49 2026

@author: tipanoii
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# =====================================================
# KONFIGURASI MODE
# =====================================================
MODES = ["seaweed", "reef"]  # proses keduanya sekaligus
def get_mode_config(mode: str):
    """
    Mengatur folder input dan output berdasarkan mode analisis.
    """

    if mode == "seaweed":
        print("\n============== KONVERSI GEOJSON RUMPUT LAUT ========================")

        input_dirs = [
            Path("D:/TA/code/seaweed/output_potensial/35m"),
            Path("D:/TA/code/seaweed/output_potensial/10m"),
        ]

        output_dir = Path("D:/TA/code/web/geojsonSeaweed")
        output_dir.mkdir(parents=True, exist_ok=True)

        return input_dirs, output_dir

    elif mode == "reef":
        print("\n============== KONVERSI GEOJSON TERUMBU KARANG ========================")

        input_dirs = [
            Path("D:/TA/code/reef/output_potensial/35m"),
            Path("D:/TA/code/reef/output_potensial/10m"),
        ]

        output_dir = Path("D:/TA/code/web/geojsonReef")
        output_dir.mkdir(parents=True, exist_ok=True)

        return input_dirs, output_dir

    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


# =====================================================
# FILTER FILE BERDASARKAN MERGE_GAP_PIXELS
# =====================================================
# Hanya file dengan MERGE_GAP_PIXELS = 25 yang diproses.
MERGE_GAP_FILTER = 25

# Format file:
# 25_{base_name}_{scenario_name}_zone.tif
# 25_{base_name}_{scenario_name}_ringkasan.csv
ZONE_PATTERN = f"{MERGE_GAP_FILTER}_*_zone.tif"
CSV_PATTERN = f"{MERGE_GAP_FILTER}_*_ringkasan.csv"

ZONE_SUFFIX = "_zone"
CSV_SUFFIX = "_ringkasan"

# Jika GeoJSON terlalu berat, aktifkan simplify
SIMPLIFY_GEOMETRY = True

# Jika CRS raster dalam meter, tolerance 10 berarti 10 meter.
# Jika bentuk polygon terlalu kasar, kecilkan menjadi 5.
SIMPLIFY_TOLERANCE = 10


# =====================================================
# FUNGSI BANTU
# =====================================================
def get_key_from_zone_file(path: Path) -> str:
    """
    Mengambil key dari file zone.

    Contoh:
    25_35m_2025_highest_check4factor_zone.tif

    menjadi:
    25_35m_2025_highest_check4factor

    Key ini harus sama dengan key pada file CSV.
    """
    stem = path.stem

    if not stem.endswith(ZONE_SUFFIX):
        raise ValueError(f"Nama file zone tidak sesuai format: {path.name}")

    key = stem[:-len(ZONE_SUFFIX)]

    if not key.startswith(f"{MERGE_GAP_FILTER}_"):
        raise ValueError(
            f"File zone bukan hasil MERGE_GAP_PIXELS {MERGE_GAP_FILTER}: {path.name}"
        )

    return key


def get_key_from_csv_file(path: Path) -> str:
    """
    Mengambil key dari file ringkasan CSV.

    Contoh:
    25_35m_2025_highest_check4factor_ringkasan.csv

    menjadi:
    25_35m_2025_highest_check4factor

    Key ini dipakai untuk mencocokkan CSV dengan zone TIF.
    """
    stem = path.stem

    if not stem.endswith(CSV_SUFFIX):
        raise ValueError(f"Nama file CSV tidak sesuai format: {path.name}")

    key = stem[:-len(CSV_SUFFIX)]

    if not key.startswith(f"{MERGE_GAP_FILTER}_"):
        raise ValueError(
            f"File CSV bukan hasil MERGE_GAP_PIXELS {MERGE_GAP_FILTER}: {path.name}"
        )

    return key


def raster_zone_to_gdf(zone_tif: Path) -> gpd.GeoDataFrame:
    """
    Mengubah raster zone_id menjadi polygon GeoDataFrame.
    Hanya pixel dengan zone_id > 0 yang diambil.
    """
    records = []

    with rasterio.open(zone_tif) as src:
        zone_arr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        if crs is None:
            raise ValueError(f"Raster tidak memiliki CRS: {zone_tif}")

        # Ambil zona valid.
        # zone_id 0 dianggap background/nodata sesuai kode sebelumnya.
        valid_mask = zone_arr > 0

        if nodata is not None:
            valid_mask = valid_mask & (zone_arr != nodata)

        for geom, value in shapes(
            zone_arr.astype("int32"),
            mask=valid_mask,
            transform=transform
        ):
            zone_id = int(value)

            if zone_id <= 0:
                continue

            records.append({
                "zone_id": zone_id,
                "geometry": shape(geom)
            })

    if len(records) == 0:
        return gpd.GeoDataFrame(columns=["zone_id", "geometry"], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Jika satu zone_id terpecah menjadi beberapa polygon,
    # gabungkan kembali berdasarkan zone_id.
    gdf = gdf.dissolve(by="zone_id", as_index=False)

    return gdf


def convert_pair_to_geojson(zone_tif: Path, csv_path: Path, output_geojson: Path, depth_label: str):
    print("\n====================================================")
    print(f"Zone TIF : {zone_tif.name}")
    print(f"CSV      : {csv_path.name}")
    print(f"Output   : {output_geojson}")
    print("====================================================")

    # 1. Raster zone_id menjadi polygon
    gdf_zone = raster_zone_to_gdf(zone_tif)

    if gdf_zone.empty:
        print(f"File dilewati karena tidak ada zona valid: {zone_tif.name}")
        return

    # 2. Baca CSV atribut
    df_attr = pd.read_csv(csv_path, sep=";")

    if "zone_id" not in df_attr.columns:
        raise ValueError(f"CSV tidak memiliki kolom zone_id: {csv_path}")

    gdf_zone["zone_id"] = gdf_zone["zone_id"].astype(int)
    df_attr["zone_id"] = df_attr["zone_id"].astype(int)

    # 3. Join atribut CSV ke polygon berdasarkan zone_id
    gdf = gdf_zone.merge(df_attr, on="zone_id", how="left")

    # 4. Tambahkan metadata supaya saat web tahu ini data 10m atau 35m
    gdf["depth_group"] = depth_label
    gdf["source_zone_tif"] = zone_tif.name
    gdf["source_csv"] = csv_path.name

    # 5. Sederhanakan geometry agar GeoJSON lebih ringan
    if SIMPLIFY_GEOMETRY and not gdf.crs.is_geographic:
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=SIMPLIFY_TOLERANCE,
            preserve_topology=True
        )

    # 6. Ubah ke EPSG:4326 agar cocok untuk web map
    gdf = gdf.to_crs(epsg=4326)

    # 7. Simpan GeoJSON
    gdf.to_file(output_geojson, driver="GeoJSON")

    print(f"Berhasil dibuat: {output_geojson.name}")
    print(f"Jumlah zona: {len(gdf)}")

def get_scenario_from_key(key: str) -> str:
    """
    Mengambil nama skenario dari key file.

    Contoh:
    25_10m_2025_highest_check4factor   -> check4factor
    25_10m_2025_highest_noCheck4factor -> noCheck4factor
    """
    if key.endswith("_noCheck4factor"):
        return "noCheck4factor"

    if key.endswith("_check4factor"):
        return "check4factor"

    raise ValueError(f"Skenario check4factor/noCheck4factor tidak ditemukan pada key: {key}")
    
def process_folder(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        print(f"Folder tidak ditemukan: {input_dir}")
        return

    depth_label = input_dir.name  # "35m" atau "10m"

    # Hanya mengambil file dengan MERGE_GAP_PIXELS = 25
    zone_files = sorted(input_dir.glob(ZONE_PATTERN))
    csv_files = sorted(input_dir.glob(CSV_PATTERN))

    print("\n====================================================")
    print(f"Memproses folder: {input_dir}")
    print(f"Filter merge gap : {MERGE_GAP_FILTER}")
    print(f"Jumlah zone TIF  : {len(zone_files)}")
    print(f"Jumlah CSV       : {len(csv_files)}")
    print("====================================================")

    csv_map = {}

    for csv_path in csv_files:
        key = get_key_from_csv_file(csv_path)
        csv_map[key] = csv_path

    success_count = 0
    missing_count = 0

    for zone_tif in zone_files:
        key = get_key_from_zone_file(zone_tif)

        if key not in csv_map:
            print(f"Tidak ada pasangan CSV untuk: {zone_tif.name}")
            missing_count += 1
            continue
        csv_path = csv_map[key]

        scenario_name = get_scenario_from_key(key)
        output_geojson = output_dir / f"zona_potensial_{depth_label}_{scenario_name}.geojson"
        
        convert_pair_to_geojson(
            zone_tif=zone_tif,
            csv_path=csv_path,
            output_geojson=output_geojson,
            depth_label=depth_label
        )
        success_count += 1

    print("\nRingkasan folder:")
    print(f"Berhasil dikonversi : {success_count}")
    print(f"Tidak ada pasangan  : {missing_count}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    for mode in MODES:
        input_dirs, output_dir = get_mode_config(mode)

        print("\n====================================================")
        print(f"Mode yang diproses: {mode}")
        print(f"Output GeoJSON    : {output_dir}")
        print("====================================================")

        for input_dir in input_dirs:
            process_folder(
                input_dir=input_dir,
                output_dir=output_dir
            )

    print("\n====================================================")
    print("SELURUH PROSES GEOJSON SELESAI")
    print("====================================================")