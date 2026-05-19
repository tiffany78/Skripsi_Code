#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:47:47 2026

@author: tipanoii
"""

from pathlib import Path
import re

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


# =====================================================
# KONFIGURASI PATH
# =====================================================
DEPTH_TIF_DIR = Path("/Users/tipanoii/doc/TA/code/depth")
DEPTH_GEOJSON_DIR = Path("/Users/tipanoii/doc/TA/code/web/geojsonDepth")

DEPTH_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

SIMPLIFY_GEOMETRY = True
SIMPLIFY_TOLERANCE = 10


# =====================================================
# FUNGSI BANTU
# =====================================================
def parse_depth_filename(tif_path: Path):
    """
    Format nama file yang diharapkan:
    Depth_10m_2023.tif
    Depth_35m_2023.tif

    Output:
    depth_group = "10m" atau "35m"
    year = "2023"
    max_depth = 10 atau 35
    """
    pattern = r"Depth_(10m|35m)_(20\d{2})$"
    match = re.search(pattern, tif_path.stem)

    if not match:
        raise ValueError(
            f"Nama file tidak sesuai format Depth_10m_2023 atau Depth_35m_2023: {tif_path.name}"
        )

    depth_group = match.group(1)
    year = match.group(2)

    if depth_group == "10m":
        max_depth = 10
    elif depth_group == "35m":
        max_depth = 35
    else:
        raise ValueError(f"Depth group tidak dikenali: {depth_group}")

    return depth_group, year, max_depth


def convert_depth_tif_to_interval_geojson(tif_path: Path, output_path: Path, max_depth: int):
    print("\n====================================================")
    print(f"Input     : {tif_path}")
    print(f"Output    : {output_path}")
    print(f"Max depth : {max_depth} m")
    print("====================================================")

    records = []

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        if crs is None:
            raise ValueError(f"GeoTIFF tidak memiliki CRS: {tif_path}")

        # =====================================================
        # MASK VALID
        # =====================================================
        valid_mask = np.isfinite(arr)

        if nodata is not None:
            valid_mask = valid_mask & (arr != nodata)

        # =====================================================
        # NORMALISASI NILAI KEDALAMAN
        # =====================================================
        # Jika nilai kedalaman negatif, misalnya -5,
        # maka diubah menjadi positif 5.
        depth_positive = np.where(arr < 0, -arr, arr)

        valid_mask = (
            valid_mask
            & np.isfinite(depth_positive)
            & (depth_positive >= 0)
            & (depth_positive <= max_depth)
        )

        # =====================================================
        # KLASIFIKASI INTERVAL 1 METER
        # =====================================================
        class_arr = np.full(arr.shape, -9999, dtype=np.int16)

        interval_values = np.floor(depth_positive[valid_mask])
        interval_values = np.clip(interval_values, 0, max_depth - 1)

        class_arr[valid_mask] = interval_values.astype(np.int16)

        # =====================================================
        # POLYGONIZE
        # =====================================================
        for geom, value in shapes(
            class_arr,
            mask=valid_mask,
            transform=transform
        ):
            lower = int(value)

            if lower < 0:
                continue

            upper = lower + 1

            records.append({
                "lower_m": lower,
                "upper_m": upper,
                "depth_interval": f"{lower}-{upper} m",
                "geometry": shape(geom)
            })

    if not records:
        print("Tidak ada data valid.")
        return

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Gabungkan polygon berdasarkan interval kedalaman
    gdf = gdf.dissolve(
        by=["lower_m", "upper_m", "depth_interval"],
        as_index=False
    )

    # Hitung area jika CRS dalam satuan meter
    if not gdf.crs.is_geographic:
        gdf["area_ha"] = (gdf.geometry.area / 10000).round(3)
    else:
        gdf["area_ha"] = None

    # Sederhanakan geometry agar file lebih ringan
    if SIMPLIFY_GEOMETRY and not gdf.crs.is_geographic:
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=SIMPLIFY_TOLERANCE,
            preserve_topology=True
        )

    # Web map menggunakan EPSG:32754
    gdf = gdf.to_crs(epsg=32754)

    gdf.to_file(output_path, driver="GeoJSON")

    print(f"Berhasil dibuat: {output_path.name}")
    print(f"Jumlah interval: {len(gdf)}")


# =====================================================
# MAIN LOOP
# =====================================================
if __name__ == "__main__":
    tif_files = sorted(DEPTH_TIF_DIR.glob("Depth_*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"Tidak ada file Depth_*.tif di folder: {DEPTH_TIF_DIR}")

    for tif_path in tif_files:
        try:
            depth_group, year, max_depth = parse_depth_filename(tif_path)

            output_name = f"Depth_{depth_group}_{year}_interval_1m.geojson"
            output_path = DEPTH_GEOJSON_DIR / output_name

            convert_depth_tif_to_interval_geojson(
                tif_path=tif_path,
                output_path=output_path,
                max_depth=max_depth
            )

        except Exception as e:
            print(f"Gagal memproses {tif_path.name}: {e}")

    print("\nSELURUH KONVERSI SELESAI")
    print(f"Output GeoJSON tersimpan di: {DEPTH_GEOJSON_DIR}")