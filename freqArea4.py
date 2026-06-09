# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:14 2026

@author: Tiffany
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
import csv
import re
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = ""
MODE = "seaweed"  # "seaweed" or "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS RUMPUT LAUT ========================")
    ROOT = Path("D:/TA/code/seaweed/output_potensial")

    sal_min = 28
    sal_max = 34

    depth_min = 3
    depth_max = 5

    temp_min = 24
    temp_max = 30

    sen_min = 0
    sen_max = 25

elif MODE == "reef":
    print("\n============== ANALISIS TERUMBU KARANG ========================")
    ROOT = Path("D:/TA/code/reef/output_potensial")

    sal_min = 30
    sal_max = 35

    depth_min = 4
    depth_max = 8

    temp_min = 23
    temp_max = 30

    sen_min = 0
    sen_max = 20

else:
    raise SystemExit("Tipe Analisis Tidak Sesuai")

# Input berasal dari hasil analisis faktor sebelumnya
INPUT_TIFS = sorted(ROOT.glob("factor_suitability_*.tif"))

# =========================================================
# PATH DATA LINGKUNGAN
# =========================================================
ROOT_SAL = Path("D:/TA/code/salinity/")
ROOT_TEMP = Path("D:/TA/code/temp/")
ROOT_DEPTH = Path("D:/TA/code/depth/")
ROOT_SEN = Path("D:/TA/code/sediment/")

SAL_VAR = "sos"
TEMP_VAR = "to"
SEN_VAR = "SPM"

TIME_MODE = "mean"      # "mean", "median", atau "first"
SURFACE_DEPTH = 0

# Nilai faktor yang dianggap potensial dari raster factor_suitability
# Nilai 4 berarti sudah memenuhi 4 faktor pada proses sebelumnya
POTENTIAL_VALUES = [4]

# =========================================================
# PARAMETER SPASIAL
# =========================================================
PIXEL_SIZE = 20  # meter
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

# =========================================================
# PARAMETER PENYARINGAN AREA
# =========================================================
MIN_AREA = 2  # hektare
MIN_PIXELS = int((MIN_AREA * 10000) / PIXEL_AREA_M2)

# =========================================================
# PARAMETER PENGGABUNGAN AREA
# =========================================================
MERGE_DISTANCE = 500  # meter
MERGE_GAP_PIXELS = int(MERGE_DISTANCE / PIXEL_SIZE)

FILL_HOLES = True
USE_MERGED_AREA_AS_FINAL = True

# Untuk membandingkan dua skenario:
# True  = piksel hasil merge/fill dicek ulang dengan 4 faktor
# False = piksel hasil merge/fill tidak dicek ulang dengan 4 faktor
CHECK_MERGED_SCENARIOS = [True, False]

# =========================================================
# VISUALISASI
# =========================================================
SHOW_PLOT = False
FIGSIZE = (10, 8)
PNG_DPI = 300
BORDER_THICKNESS = -1

# 0 = background
# 1 = area final hasil merge/fill
# 2 = core area, yaitu piksel yang sejak awal bernilai 4
CLASS_COLORS = np.array([
    [0.00, 0.00, 0.00, 1.0],
    [0.72, 0.90, 0.45, 1.0],
    [0.18, 0.65, 0.22, 1.0],
], dtype=float)

BORDER_COLOR = [1, 1, 1, 1]

# Nodata output
# Untuk mask dan class, 255 dipakai agar 0 tetap bisa bermakna background valid
MASK_NODATA = 255
ZONE_NODATA = 0
CLASS_NODATA = 255

# Struktur 8-neighbor
STRUCTURE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)


# =========================================================
# FUNGSI BANTU
# =========================================================
def build_outline(mask, thickness):
    if thickness <= 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = ndimage.binary_erosion(mask, structure=STRUCTURE, iterations=1)
    edge = mask & (~eroded)

    if thickness > 1:
        edge = ndimage.binary_dilation(
            edge,
            structure=STRUCTURE,
            iterations=thickness - 1
        )

    return edge


def get_depth_suffix(input_tif: Path) -> str:
    """
    Mengambil suffix depth dari nama file.

    Contoh:
    factor_suitability_2025_Depth_35m_2025_highest.tif -> 35m
    factor_suitability_2025_Depth_10m_2025_highest.tif -> 10m
    """
    stem = input_tif.stem
    match = re.search(r"Depth_(10m|35m)", stem)

    if not match:
        raise ValueError(
            f"Depth suffix 10m/35m tidak ditemukan pada nama file: {input_tif.name}"
        )

    return match.group(1)


def get_depth_tif_for_input(input_tif: Path) -> Path:
    """
    Mapping input factor_suitability ke raster depth yang sesuai.
    """
    depth_suffix = get_depth_suffix(input_tif)

    depth_map = {
        "35m": ROOT_DEPTH / "Depth_35m_2025.tif",
        "10m": ROOT_DEPTH / "Depth_10m_2025.tif",
    }

    if depth_suffix not in depth_map:
        raise FileNotFoundError(
            f"Tidak ada mapping depth untuk suffix '{depth_suffix}' "
            f"pada file {input_tif.name}"
        )

    depth_tif = depth_map[depth_suffix]

    if not depth_tif.exists():
        raise FileNotFoundError(f"File depth tidak ditemukan: {depth_tif}")

    return depth_tif


def safe_name(text: str) -> str:
    """
    Membuat nama file output lebih pendek dan aman.
    """
    text = re.sub(r"^factor_suitability_\d{4}_Depth_", "", text)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def extract_year_from_name(name: str):
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else None


def standardize_xy(da):
    """
    Menyeragamkan nama dimensi spasial menjadi x dan y.
    """
    rename_dict = {}

    if "longitude" in da.dims:
        rename_dict["longitude"] = "x"
    elif "lon" in da.dims:
        rename_dict["lon"] = "x"

    if "latitude" in da.dims:
        rename_dict["latitude"] = "y"
    elif "lat" in da.dims:
        rename_dict["lat"] = "y"

    da = da.rename(rename_dict)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da


def reduce_time(da, year):
    """
    Mengambil data sesuai tahun dan mereduksi dimensi waktu.
    """
    if "time" not in da.dims:
        return da

    da_year = da.sel(time=slice(f"{year}-01-01", f"{int(year) + 1}-01-01"))

    if TIME_MODE == "mean":
        return da_year.mean("time", skipna=True)
    elif TIME_MODE == "median":
        return da_year.median("time", skipna=True)
    elif TIME_MODE == "first":
        return da_year.isel(time=0)
    else:
        raise ValueError("TIME_MODE harus 'mean', 'median', atau 'first'")


def reduce_surface(da):
    """
    Mengambil layer permukaan jika data memiliki dimensi kedalaman.
    """
    if "depth" in da.dims:
        return da.sel(depth=SURFACE_DEPTH, method="nearest")
    elif "deptht" in da.dims:
        return da.sel(deptht=SURFACE_DEPTH, method="nearest")
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


def build_factor4_mask(sal_arr, depth_arr, temp_arr, sen_arr):
    """
    Mengecek piksel yang lolos 4 faktor:
    salinitas, kedalaman, suhu, dan sedimentasi.
    """
    depth_abs = np.abs(depth_arr)

    sal_ok = (
        np.isfinite(sal_arr)
        & (sal_arr >= sal_min)
        & (sal_arr <= sal_max)
    )

    depth_ok = (
        np.isfinite(depth_abs)
        & (depth_abs >= depth_min)
        & (depth_abs <= depth_max)
    )

    temp_ok = (
        np.isfinite(temp_arr)
        & (temp_arr >= temp_min)
        & (temp_arr <= temp_max)
    )

    sen_ok = (
        np.isfinite(sen_arr)
        & (sen_arr >= sen_min)
        & (sen_arr <= sen_max)
    )

    return sal_ok & depth_ok & temp_ok & sen_ok


def load_factor_layers(year, zone_grid, depth_tif):
    """
    Memuat salinitas, suhu, sedimentasi, dan kedalaman,
    lalu menyamakan grid-nya dengan raster input.
    """
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

    sal_on_zone = cur_sal.rio.reproject_match(
        zone_grid,
        resampling=Resampling.nearest
    )

    temp_on_zone = cur_temp.rio.reproject_match(
        zone_grid,
        resampling=Resampling.nearest
    )

    sen_on_zone = cur_sen.rio.reproject_match(
        zone_grid,
        resampling=Resampling.nearest
    )

    ds_sal.close()
    ds_temp.close()
    ds_sen.close()

    depth_da = rxr.open_rasterio(depth_tif, masked=True).squeeze(drop=True)

    if depth_da.rio.crs is None:
        raise ValueError(f"Raster depth tidak punya CRS: {depth_tif}")

    depth_on_zone = depth_da.rio.reproject_match(
        zone_grid,
        resampling=Resampling.nearest
    )

    return (
        np.asarray(sal_on_zone.values, dtype=float),
        np.abs(np.asarray(depth_on_zone.values, dtype=float)),
        np.asarray(temp_on_zone.values, dtype=float),
        np.asarray(sen_on_zone.values, dtype=float),
    )


def create_output_paths(output_dir, base_name, min_area_str, scenario_name):
    """
    Membuat path output untuk setiap skenario.
    """
    common = f"{base_name}_{min_area_str}_{MERGE_GAP_PIXELS}_{scenario_name}"

    return {
        "mask_tif": output_dir / f"mask_{common}.tif",
        "zone_tif": output_dir / f"zone_{common}.tif",
        "class_tif": output_dir / f"class_{common}.tif",
        "png": output_dir / f"peta_{common}.png",
        "csv": output_dir / f"ringkasan_{common}.csv",
    }


def process_scenario(
    input_tif,
    arr,
    profile,
    valid_mask,
    candidate_mask,
    base_merged_mask,
    factor4_mask,
    sal_arr,
    depth_arr,
    temp_arr,
    sen_arr,
    output_dir,
    base_name,
    check_merged_pixels_with_4_factors
):
    """
    Memproses satu skenario:
    - check_merged_pixels_with_4_factors = True
      hasil merge/fill dicek ulang dengan 4 faktor.

    - check_merged_pixels_with_4_factors = False
      hasil merge/fill langsung dipakai tanpa cek ulang 4 faktor.
    """
    scenario_name = (
        "check4factor"
        if check_merged_pixels_with_4_factors
        else "noCheck4factor"
    )

    min_area_str = str(MIN_AREA)
    paths = create_output_paths(
        output_dir=output_dir,
        base_name=base_name,
        min_area_str=min_area_str,
        scenario_name=scenario_name
    )

    print("\n----------------------------------------------------")
    print(f"Skenario: {scenario_name}")
    print("----------------------------------------------------")

    if USE_MERGED_AREA_AS_FINAL and check_merged_pixels_with_4_factors:
        # Piksel kandidat awal tetap dipertahankan karena sudah berasal dari factor_suitability = 4.
        # Piksel tambahan hasil merge/fill saja yang dicek ulang dengan 4 faktor.
        added_mask = base_merged_mask & (~candidate_mask)
    
        working_mask = candidate_mask | (added_mask & factor4_mask)
    
        # Tetap batasi ke area valid input
        working_mask = working_mask & valid_mask
    else:
        working_mask = base_merged_mask & valid_mask

    print(
        "Jumlah piksel setelah merge/fill pada skenario ini:",
        int(np.count_nonzero(working_mask))
    )

    labeled, num_features = ndimage.label(working_mask, structure=STRUCTURE)
    component_sizes = np.bincount(labeled.ravel())

    print("Jumlah komponen hasil merge:", num_features)

    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    zone_arr = np.zeros_like(labeled, dtype=np.int32)

    zone_stats = []
    new_id = 1

    for comp_id in range(1, len(component_sizes)):
        comp_mask = labeled == comp_id

        if USE_MERGED_AREA_AS_FINAL:
            comp_final = comp_mask
        else:
            comp_final = comp_mask & candidate_mask

        pixel_count = int(np.count_nonzero(comp_final))

        if pixel_count < MIN_PIXELS:
            continue

        final_mask |= comp_final
        zone_arr[comp_final] = new_id

        area_m2 = pixel_count * PIXEL_AREA_M2
        area_ha = area_m2 / 10000.0

        zone_stats.append({
            "zone_id": new_id,
            "pixel_count": pixel_count,
            "area_m2": round(area_m2, 3),
            "area_ha": round(area_ha, 3),
        })

        new_id += 1

    print("\nRingkasan zona akhir:")
    if len(zone_stats) == 0:
        print("Tidak ada zona yang lolos.")
    else:
        print(f"Total zona: {new_id - 1}")

        csv_rows = []

        for z in zone_stats:
            zid = z["zone_id"]
            mask = zone_arr == zid

            row = {
                "scenario": scenario_name,
                "zone_id": zid,
                "pixel_count": z["pixel_count"],
                "area_m2": z["area_m2"],
                "area_ha": z["area_ha"],
                "input_file": input_tif.name,
            }

            row.update(summarize_values(sal_arr[mask], "salinity"))
            row.update(summarize_values(depth_arr[mask], "depth"))
            row.update(summarize_values(temp_arr[mask], "temperature"))
            row.update(summarize_values(sen_arr[mask], "sedimentation"))

            csv_rows.append(row)

        fieldnames = [
            "scenario",
            "zone_id",
            "pixel_count",
            "area_m2",
            "area_ha",
            "input_file",
            "salinity_min",
            "salinity_max",
            "salinity_mean",
            "salinity_median",
            "depth_min",
            "depth_max",
            "depth_mean",
            "depth_median",
            "temperature_min",
            "temperature_max",
            "temperature_mean",
            "temperature_median",
            "sedimentation_min",
            "sedimentation_max",
            "sedimentation_mean",
            "sedimentation_median",
        ]

        with open(paths["csv"], mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Ringkasan CSV tersimpan: {paths['csv']}")

    # =====================================================
    # BUAT CLASS RASTER
    # 255 = nodata
    # 0   = background valid
    # 1   = area final hasil merge/fill
    # 2   = core area, yaitu piksel kandidat awal bernilai 4
    # =====================================================
    class_arr = np.full(arr.shape, CLASS_NODATA, dtype=np.uint8)
    class_arr[valid_mask] = 0
    class_arr[final_mask] = 1

    highest_value = max(POTENTIAL_VALUES)
    core_mask = valid_mask & (arr == highest_value) & final_mask
    class_arr[core_mask] = 2

    print("Jumlah piksel final:", int(np.count_nonzero(final_mask)))
    print("Jumlah piksel core :", int(np.count_nonzero(core_mask)))

    # =====================================================
    # SIMPAN MASK
    # =====================================================
    mask_out = np.full(arr.shape, MASK_NODATA, dtype=np.uint8)
    mask_out[valid_mask] = 0
    mask_out[final_mask] = 1

    mask_profile = profile.copy()
    mask_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=MASK_NODATA,
        compress="lzw"
    )

    with rasterio.open(paths["mask_tif"], "w", **mask_profile) as dst:
        dst.write(mask_out, 1)

    print(f"Mask final tersimpan: {paths['mask_tif']}")

    # =====================================================
    # SIMPAN ZONE ID
    # =====================================================
    zone_profile = profile.copy()
    zone_profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=ZONE_NODATA,
        compress="lzw"
    )

    with rasterio.open(paths["zone_tif"], "w", **zone_profile) as dst:
        dst.write(zone_arr, 1)

    print(f"Zone ID tersimpan: {paths['zone_tif']}")

    # =====================================================
    # SIMPAN CLASS RASTER
    # =====================================================
    class_profile = profile.copy()
    class_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=CLASS_NODATA,
        compress="lzw"
    )

    with rasterio.open(paths["class_tif"], "w", **class_profile) as dst:
        dst.write(class_arr, 1)

    print(f"Class raster tersimpan: {paths['class_tif']}")

    # =====================================================
    # SIMPAN PNG
    # =====================================================
    display_arr = class_arr.copy()
    display_arr[display_arr == CLASS_NODATA] = 0

    cmap = ListedColormap(CLASS_COLORS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(display_arr, cmap=cmap, interpolation="nearest")

    if BORDER_THICKNESS > 0:
        outline_mask = build_outline(final_mask, thickness=BORDER_THICKNESS)
        outline_rgba = np.zeros(
            (final_mask.shape[0], final_mask.shape[1], 4),
            dtype=float
        )
        outline_rgba[outline_mask] = BORDER_COLOR
        ax.imshow(outline_rgba, interpolation="nearest")

    ax.set_title(
        "Peta Zona Potensial\n"
        f"Input={input_tif.name} | "
        f"Skenario={scenario_name} | "
        f"Potential={POTENTIAL_VALUES} | "
        f"Min area={MIN_PIXELS} px | "
        f"Merge gap={MERGE_GAP_PIXELS} px "
        f"({MERGE_GAP_PIXELS * PIXEL_SIZE} m) | "
        f"Fill holes={FILL_HOLES}"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(paths["png"], dpi=PNG_DPI, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG peta tersimpan: {paths['png']}")


def process_one_tif(input_tif: Path):
    depth_suffix = get_depth_suffix(input_tif)
    output_dir = ROOT / depth_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_name(input_tif.stem)

    print("\n====================================================")
    print(f"Memproses : {input_tif.name}")
    print(f"Folder out: {output_dir}")
    print("====================================================")

    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        input_nodata = src.nodata

    print("Shape:", arr.shape)
    print("CRS:", profile.get("crs"))
    print("Input nodata:", input_nodata)

    if input_nodata is not None:
        valid_mask = arr != input_nodata
    else:
        valid_mask = np.isfinite(arr)

    candidate_mask = valid_mask & np.isin(arr, POTENTIAL_VALUES)

    print("Jumlah piksel kandidat awal:", int(np.count_nonzero(candidate_mask)))

    # =====================================================
    # LOAD 4 FAKTOR
    # Dibutuhkan untuk skenario check4factor dan ringkasan CSV.
    # =====================================================
    year = extract_year_from_name(input_tif.stem)
    if year is None:
        raise ValueError(f"Tahun tidak ditemukan pada nama file: {input_tif.name}")

    zone_grid = rxr.open_rasterio(input_tif, masked=True).squeeze(drop=True)

    depth_tif = get_depth_tif_for_input(input_tif)
    print(f"Depth yang dipakai: {depth_tif.name}")

    sal_arr, depth_arr, temp_arr, sen_arr = load_factor_layers(
        year=year,
        zone_grid=zone_grid,
        depth_tif=depth_tif
    )

    factor4_mask = build_factor4_mask(
        sal_arr=sal_arr,
        depth_arr=depth_arr,
        temp_arr=temp_arr,
        sen_arr=sen_arr
    )

    print(
        "Jumlah piksel yang lolos 4 faktor:",
        int(np.count_nonzero(factor4_mask))
    )

    # =====================================================
    # MERGE/FILL AREA DASAR
    # Hasil ini menjadi dasar perbandingan dua skenario.
    # =====================================================
    base_merged_mask = candidate_mask.copy()

    if MERGE_GAP_PIXELS > 0:
        base_merged_mask = ndimage.binary_closing(
            base_merged_mask,
            structure=STRUCTURE,
            iterations=MERGE_GAP_PIXELS
        )

    if FILL_HOLES:
        base_merged_mask = ndimage.binary_fill_holes(base_merged_mask)

    # Menjaga agar hasil merge/fill tidak masuk ke area nodata input
    base_merged_mask = base_merged_mask & valid_mask

    print(
        "Jumlah piksel setelah merge/fill dasar:",
        int(np.count_nonzero(base_merged_mask))
    )

    # =====================================================
    # PROSES DUA SKENARIO
    # =====================================================
    for check_merged in CHECK_MERGED_SCENARIOS:
        process_scenario(
            input_tif=input_tif,
            arr=arr,
            profile=profile,
            valid_mask=valid_mask,
            candidate_mask=candidate_mask,
            base_merged_mask=base_merged_mask,
            factor4_mask=factor4_mask,
            sal_arr=sal_arr,
            depth_arr=depth_arr,
            temp_arr=temp_arr,
            sen_arr=sen_arr,
            output_dir=output_dir,
            base_name=base_name,
            check_merged_pixels_with_4_factors=check_merged
        )


# =========================================================
# MAIN LOOP
# =========================================================
if not INPUT_TIFS:
    raise FileNotFoundError(
        f"Tidak ada file factor_suitability_*.tif di folder: {ROOT}"
    )

print("Jumlah file input ditemukan:", len(INPUT_TIFS))
for tif in INPUT_TIFS:
    try:
        process_one_tif(tif)
    except Exception as e:
        print(f"Gagal memproses {tif.name}: {e}")