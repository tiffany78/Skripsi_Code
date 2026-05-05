#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:59:36 2026

@author: tipanoii
"""

from pathlib import Path
import re
import numpy as np
import csv
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

# =========================================================
# KONFIGURASI
# =========================================================
MODE = "reef"   # "seaweed" or "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS EKSTENSI ZONA RUMPUT LAUT ========================")
    ROOT_BASE = Path("/Users/tipanoii/doc/TA/code/seaweed")

    # batas faktor eksternal rumput laut
    sal_min, sal_max = 28, 34
    depth_min, depth_max = 3, 5
    temp_min, temp_max = 24, 30
    sen_min, sen_max = 0, 25

elif MODE == "reef":
    print("\n============== ANALISIS EKSTENSI ZONA TERUMBU KARANG ========================")
    ROOT_BASE = Path("/Users/tipanoii/doc/TA/code/reef")

    # batas faktor eksternal terumbu karang
    sal_min, sal_max = 30, 35
    depth_min, depth_max = 4, 8
    temp_min, temp_max = 23, 30
    sen_min, sen_max = 0, 20
else:
    raise SystemExit("MODE harus 'seaweed' atau 'reef'")

# input zona mask hasil sebelumnya
ROOT_ZONE = ROOT_BASE / "output_filtering"

# file input: subfolder 10m / 35m
INPUT_ZONE_MASKS = sorted(ROOT_ZONE.glob("*/zona_potensial_mask_*.tif"))

# 4 faktor eksternal
ROOT_SAL   = Path("/Users/tipanoii/doc/TA/code/salinity")
ROOT_TEMP  = Path("/Users/tipanoii/doc/TA/code/temp")
ROOT_SEN   = Path("/Users/tipanoii/doc/TA/code/sediment")
ROOT_DEPTH = Path("/Users/tipanoii/doc/TA/code/depth")

SAL_VAR = "sos"
TEMP_VAR = "to"
SEN_VAR = "SPM"

TIME_MODE = "mean"   # "mean", "median", "first"
SURFACE_DEPTH = 0

# resolusi piksel
PIXEL_SIZE = 20
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

# neighbor
USE_8_NEIGHBOR = True
STRUCTURE_8 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)

STRUCTURE_4 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

STRUCTURE = STRUCTURE_8 if USE_8_NEIGHBOR else STRUCTURE_4

# visualisasi
SHOW_PLOT = False
FIGSIZE = (10, 8)
PNG_DPI = 300

OUT_NODATA = 0


# =========================================================
# HELPER
# =========================================================
def extract_year_from_name(name: str):
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else None


def standardize_xy(da):
    rename_map = {}

    if "longitude" in da.dims:
        rename_map["longitude"] = "x"
    elif "lon" in da.dims:
        rename_map["lon"] = "x"

    if "latitude" in da.dims:
        rename_map["latitude"] = "y"
    elif "lat" in da.dims:
        rename_map["lat"] = "y"

    da = da.rename(rename_map)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da


def reduce_time(da, year):
    if "time" not in da.dims:
        return da

    da_year = da.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01"))

    if TIME_MODE == "mean":
        return da_year.mean("time", skipna=True)
    elif TIME_MODE == "median":
        return da_year.median("time", skipna=True)
    elif TIME_MODE == "first":
        return da_year.isel(time=0)
    else:
        raise ValueError("TIME_MODE harus 'mean', 'median', atau 'first'")


def reduce_surface(da):
    if "depth" in da.dims:
        da = da.sel(depth=SURFACE_DEPTH, method="nearest")
    elif "deptht" in da.dims:
        da = da.sel(deptht=SURFACE_DEPTH, method="nearest")
    return da


def summarize_values(values, prefix):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
        }

    return {
        f"{prefix}_min": round(float(np.min(arr)), 3),
        f"{prefix}_max": round(float(np.max(arr)), 3),
        f"{prefix}_mean": round(float(np.mean(arr)), 3),
        f"{prefix}_median": round(float(np.median(arr)), 3),
    }


def get_depth_tif_from_folder(zone_mask_tif: Path) -> Path:
    """
    Ambil depth berdasarkan nama folder induk:
    output_filtering/35m/zona_potensial_mask_...
    output_filtering/10m/zona_potensial_mask_...
    """
    depth_suffix = zone_mask_tif.parent.name

    depth_map = {
        "35m": ROOT_DEPTH / "Depth_Batnas_Focal_35m.tif",
        "10m": ROOT_DEPTH / "Depth_Batnas_10m.tif",
    }

    if depth_suffix not in depth_map:
        raise FileNotFoundError(
            f"Tidak ada mapping depth untuk folder '{depth_suffix}' pada file {zone_mask_tif.name}"
        )

    depth_tif = depth_map[depth_suffix]

    if not depth_tif.exists():
        raise FileNotFoundError(f"File depth tidak ditemukan: {depth_tif}")

    return depth_tif


def get_zone_id_tif_from_mask(zone_mask_tif: Path) -> Path:
    """
    Cari file zone id yang sesuai:
    zona_potensial_mask_xxx.tif -> zona_potensial_zone_xxx.tif
    """
    zone_id_name = zone_mask_tif.name.replace("zona_potensial_mask_", "zona_potensial_zone_", 1)
    return zone_mask_tif.parent / zone_id_name


def load_factor_layers(year, zone_grid, depth_tif):
    sal_nc = ROOT_SAL / f"{year}.nc"
    temp_nc = ROOT_TEMP / f"{year}.nc"
    sen_nc = ROOT_SEN / f"{year}.nc"

    if not sal_nc.exists():
        raise FileNotFoundError(f"File salinitas tidak ditemukan: {sal_nc}")
    if not temp_nc.exists():
        raise FileNotFoundError(f"File suhu tidak ditemukan: {temp_nc}")
    if not sen_nc.exists():
        raise FileNotFoundError(f"File sedimentasi tidak ditemukan: {sen_nc}")
    if not depth_tif.exists():
        raise FileNotFoundError(f"File kedalaman tidak ditemukan: {depth_tif}")

    ds_sal = xr.open_dataset(sal_nc, engine="netcdf4")
    ds_temp = xr.open_dataset(temp_nc, engine="netcdf4")
    ds_sen = xr.open_dataset(sen_nc, engine="netcdf4")

    sal = ds_sal[SAL_VAR]
    temp = ds_temp[TEMP_VAR]
    sen = ds_sen[SEN_VAR]

    cur_sal = standardize_xy(reduce_surface(reduce_time(sal, year)))
    cur_temp = standardize_xy(reduce_surface(reduce_time(temp, year)))
    cur_sen = standardize_xy(reduce_surface(reduce_time(sen, year)))

    sal_on_zone = cur_sal.rio.reproject_match(zone_grid, resampling=Resampling.nearest)
    temp_on_zone = cur_temp.rio.reproject_match(zone_grid, resampling=Resampling.nearest)
    sen_on_zone = cur_sen.rio.reproject_match(zone_grid, resampling=Resampling.nearest)

    ds_sal.close()
    ds_temp.close()
    ds_sen.close()

    depth_da = rxr.open_rasterio(depth_tif, masked=True).squeeze(drop=True)
    if depth_da.rio.crs is None:
        raise ValueError(f"Raster depth tidak punya CRS: {depth_tif}")

    depth_on_zone = depth_da.rio.reproject_match(zone_grid, resampling=Resampling.nearest)

    # konsisten dengan analisis sebelumnya: depth dipakai sebagai nilai absolut
    depth_arr = np.abs(np.asarray(depth_on_zone.values, dtype=float))

    return (
        np.asarray(sal_on_zone.values, dtype=float),
        depth_arr,
        np.asarray(temp_on_zone.values, dtype=float),
        np.asarray(sen_on_zone.values, dtype=float),
    )


def save_png(class_arr, out_png, title):
    """
    class:
    0 = lainnya
    1 = zona awal
    2 = kandidat ekstensi
    """
    h, w = class_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[class_arr == 0] = [0, 0, 0]
    rgb[class_arr == 1] = [0, 153, 0]
    rgb[class_arr == 2] = [255, 153, 0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array([0, 153, 0]) / 255.0, edgecolor="black", label="zona potensial awal"),
        Patch(facecolor=np.array([255, 153, 0]) / 255.0, edgecolor="black", label="piksel 0 bertetangga + memenuhi 4 faktor"),
        Patch(facecolor=np.array([0, 0, 0]) / 255.0, edgecolor="black", label="lainnya"),
    ]

    ax.legend(
        handles=legend_elements,
        title="Kelas",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG tersimpan: {out_png}")


# =========================================================
# PROSES UTAMA
# =========================================================
def process_one_zone_mask(zone_mask_tif: Path):
    year = extract_year_from_name(zone_mask_tif.stem)
    if year is None:
        raise ValueError(f"Tahun tidak ditemukan pada nama file: {zone_mask_tif.name}")

    depth_tif = get_depth_tif_from_folder(zone_mask_tif)
    zone_id_tif = get_zone_id_tif_from_mask(zone_mask_tif)

    output_dir = zone_mask_tif.parent / "zona_ekstensi"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = zone_mask_tif.stem.replace("zona_potensial_mask_", "")

    OUT_CAND_MASK_TIF = output_dir / f"zona_ekstensi_kandidat_mask_{base_name}.tif"
    OUT_CLASS_TIF     = output_dir / f"zona_ekstensi_class_{base_name}.tif"
    OUT_ASSIGN_TIF    = output_dir / f"zona_ekstensi_assign_zone_{base_name}.tif"
    OUT_MERGED_TIF    = output_dir / f"zona_ekstensi_merged_mask_{base_name}.tif"
    OUT_CSV           = output_dir / f"zona_ekstensi_ringkasan_{base_name}.csv"
    OUT_PNG           = output_dir / f"zona_ekstensi_peta_{base_name}.png"

    print("\n====================================================")
    print(f"Memproses zone mask : {zone_mask_tif.name}")
    print(f"Depth yang dipakai  : {depth_tif.name}")
    print("====================================================")

    # =====================================================
    # BACA ZONE MASK 0/1
    # =====================================================
    with rasterio.open(zone_mask_tif) as src:
        zone_mask_arr = src.read(1)
        profile = src.profile.copy()
        zone_mask_nodata = src.nodata

    if zone_mask_nodata is not None:
        zone_mask = zone_mask_arr == 1
    else:
        zone_mask = zone_mask_arr == 1

    if np.count_nonzero(zone_mask) == 0:
        print("Tidak ada zona potensial awal. Skip.")
        return

    # =====================================================
    # BACA ZONE ID, kalau tidak ada buat dari mask
    # =====================================================
    if zone_id_tif.exists():
        with rasterio.open(zone_id_tif) as src:
            zone_arr = src.read(1).astype(np.int32)
    else:
        labeled, _ = ndimage.label(zone_mask, structure=STRUCTURE)
        zone_arr = labeled.astype(np.int32)

    # =====================================================
    # LOAD 4 FAKTOR KE GRID MASK
    # =====================================================
    zone_grid = rxr.open_rasterio(zone_mask_tif, masked=True).squeeze(drop=True)
    sal_arr, depth_arr, temp_arr, sen_arr = load_factor_layers(year, zone_grid, depth_tif)

    # =====================================================
    # HITUNG PIXEL 0 YANG ADJACENT KE 1
    # =====================================================
    dilated_zone = ndimage.binary_dilation(zone_mask, structure=STRUCTURE, iterations=1)
    adjacent_zero_mask = (~zone_mask) & dilated_zone

    # =====================================================
    # CEK 4 FAKTOR EKSTERNAL LANGSUNG
    # =====================================================
    sal_ok = np.isfinite(sal_arr) & (sal_arr >= sal_min) & (sal_arr <= sal_max)
    depth_ok = np.isfinite(depth_arr) & (depth_arr >= depth_min) & (depth_arr <= depth_max)
    temp_ok = np.isfinite(temp_arr) & (temp_arr >= temp_min) & (temp_arr <= temp_max)
    sen_ok = np.isfinite(sen_arr) & (sen_arr >= sen_min) & (sen_arr <= sen_max)

    all_four_ok = sal_ok & depth_ok & temp_ok & sen_ok

    extension_candidate_mask = adjacent_zero_mask & all_four_ok

    print("Jumlah piksel zona awal              :", int(np.count_nonzero(zone_mask)))
    print("Jumlah piksel 0 bertetangga dengan 1 :", int(np.count_nonzero(adjacent_zero_mask)))
    print("Jumlah kandidat ekstensi            :", int(np.count_nonzero(extension_candidate_mask)))

    # =====================================================
    # ASSIGN KANDIDAT KE ZONA TERDEKAT
    # =====================================================
    dist, indices = ndimage.distance_transform_edt(
        ~zone_mask, return_indices=True
    )
    nearest_zone_ids = zone_arr[indices[0], indices[1]]

    extension_zone_arr = np.zeros_like(zone_arr, dtype=np.int32)
    extension_zone_arr[extension_candidate_mask] = nearest_zone_ids[extension_candidate_mask]

    merged_mask = zone_mask | extension_candidate_mask

    # =====================================================
    # CSV RINGKASAN PER ZONA
    # =====================================================
    zone_ids = sorted([int(z) for z in np.unique(zone_arr) if z > 0])

    csv_rows = []
    for zid in zone_ids:
        original_zone_mask = zone_arr == zid
        ext_mask = extension_zone_arr == zid

        original_pixels = int(np.count_nonzero(original_zone_mask))
        extension_pixels = int(np.count_nonzero(ext_mask))
        merged_pixels = int(np.count_nonzero(original_zone_mask | ext_mask))

        row = {
            "zone_id": zid,
            "original_pixel_count": original_pixels,
            "extension_pixel_count": extension_pixels,
            "original_area_ha": round(original_pixels * PIXEL_AREA_M2 / 10000.0, 3),
            "extension_area_ha": round(extension_pixels * PIXEL_AREA_M2 / 10000.0, 3),
            "merged_area_ha": round(merged_pixels * PIXEL_AREA_M2 / 10000.0, 3),
        }

        row.update(summarize_values(sal_arr[ext_mask], "salinity"))
        row.update(summarize_values(depth_arr[ext_mask], "depth"))
        row.update(summarize_values(temp_arr[ext_mask], "temperature"))
        row.update(summarize_values(sen_arr[ext_mask], "sedimentation"))

        csv_rows.append(row)

    fieldnames = [
        "zone_id",
        "original_pixel_count",
        "extension_pixel_count",
        "original_area_ha",
        "extension_area_ha",
        "merged_area_ha",
        "salinity_min", "salinity_max", "salinity_mean", "salinity_median",
        "depth_min", "depth_max", "depth_mean", "depth_median",
        "temperature_min", "temperature_max", "temperature_mean", "temperature_median",
        "sedimentation_min", "sedimentation_max", "sedimentation_mean", "sedimentation_median",
    ]

    with open(OUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"CSV ringkasan tersimpan: {OUT_CSV}")

    # =====================================================
    # SIMPAN MASK KANDIDAT EKSTENSI
    # =====================================================
    cand_profile = profile.copy()
    cand_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=OUT_NODATA,
        compress="lzw"
    )
    with rasterio.open(OUT_CAND_MASK_TIF, "w", **cand_profile) as dst:
        dst.write(extension_candidate_mask.astype(np.uint8), 1)

    print(f"Mask kandidat ekstensi tersimpan: {OUT_CAND_MASK_TIF}")

    # =====================================================
    # SIMPAN CLASS RASTER
    # 0 = lain
    # 1 = zona awal
    # 2 = kandidat ekstensi
    # =====================================================
    class_arr = np.zeros_like(zone_arr, dtype=np.uint8)
    class_arr[zone_mask] = 1
    class_arr[extension_candidate_mask] = 2

    class_profile = profile.copy()
    class_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=OUT_NODATA,
        compress="lzw"
    )
    with rasterio.open(OUT_CLASS_TIF, "w", **class_profile) as dst:
        dst.write(class_arr, 1)

    print(f"Class raster tersimpan: {OUT_CLASS_TIF}")

    # =====================================================
    # SIMPAN ASSIGN ZONE RASTER
    # berisi zone_id untuk piksel tambahan
    # =====================================================
    assign_profile = profile.copy()
    assign_profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=OUT_NODATA,
        compress="lzw"
    )
    with rasterio.open(OUT_ASSIGN_TIF, "w", **assign_profile) as dst:
        dst.write(extension_zone_arr.astype(np.int32), 1)

    print(f"Assign zone raster tersimpan: {OUT_ASSIGN_TIF}")

    # =====================================================
    # SIMPAN MERGED MASK
    # =====================================================
    merged_profile = profile.copy()
    merged_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=OUT_NODATA,
        compress="lzw"
    )
    with rasterio.open(OUT_MERGED_TIF, "w", **merged_profile) as dst:
        dst.write(merged_mask.astype(np.uint8), 1)

    print(f"Merged mask tersimpan: {OUT_MERGED_TIF}")

    # =====================================================
    # PNG
    # =====================================================
    save_png(
        class_arr=class_arr,
        out_png=OUT_PNG,
        title=(
            f"Analisis Ekstensi Zona\n"
            f"{zone_mask_tif.name}\n"
            f"piksel 0 bertetangga dengan 1 + memenuhi 4 faktor"
        )
    )


# =========================================================
# MAIN
# =========================================================
if not INPUT_ZONE_MASKS:
    raise FileNotFoundError(
        f"Tidak ada file zona_potensial_mask_*.tif di folder bertingkat: {ROOT_ZONE}"
    )

print("Jumlah file zona mask ditemukan:", len(INPUT_ZONE_MASKS))
for f in INPUT_ZONE_MASKS:
    print(" -", f)

for zone_mask_tif in INPUT_ZONE_MASKS:
    try:
        process_one_zone_mask(zone_mask_tif)
    except Exception as e:
        print(f"Gagal memproses {zone_mask_tif.name}: {e}")