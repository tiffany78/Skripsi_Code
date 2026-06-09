# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:02:16 2026

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
MODE = "seaweed"  # "seaweed" atau "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS RUMPUT LAUT ========================")
    ROOT = Path("D:/TA/code/seaweed/output_potensial")

    # Batas kesesuaian habitat rumput laut
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

    # Batas kesesuaian habitat terumbu karang
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


# =========================================================
# INPUT
# =========================================================
# Input berupa raster hasil analisis faktor.
# Contoh nama:
# factor_suitability_2025_Depth_35m_2025_highest.tif
# factor_suitability_2025_Depth_10m_2025_highest.tif
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

# =========================================================
# PARAMETER ANALISIS
# =========================================================
# Nilai 4 berarti piksel pada raster factor_suitability
# sudah memenuhi 4 faktor pada proses sebelumnya.
POTENTIAL_VALUES = [4]

# Resolusi piksel raster
PIXEL_SIZE = 20  # meter
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

# Luas minimum zona
MIN_AREA = 2  # hektare
MIN_PIXELS = int((MIN_AREA * 10000) / PIXEL_AREA_M2)

# Jarak penggabungan area
MERGE_DISTANCE = 500  # meter (20, 100, 300, 500)
MERGE_GAP_PIXELS = int(MERGE_DISTANCE / PIXEL_SIZE)

# Fill holes tetap digunakan sesuai tujuan:
# mengisi lubang/celah internal pada area hasil penggabungan.
FILL_HOLES = True

# Jika True, area hasil merge/fill digunakan sebagai area final sementara.
# Jika False, merge hanya membantu pengelompokan, tetapi output final
# tetap hanya piksel kandidat awal.
USE_MERGED_AREA_AS_FINAL = True

# Dua skenario pembanding:
# True  = area hasil merge/fill dicek ulang dengan 4 faktor.
# False = area hasil merge/fill tidak dicek ulang dengan 4 faktor.
CHECK_MERGED_SCENARIOS = [True, False]


# =========================================================
# VISUALISASI
# =========================================================
SHOW_PLOT = False
FIGSIZE = (10, 8)
PNG_DPI = 300
BORDER_THICKNESS = -1

# Kelas visual:
# 0 = background
# 1 = area final hasil merge/fill
# 2 = core area, yaitu piksel kandidat awal bernilai 4
CLASS_COLORS = np.array([
    [0.00, 0.00, 0.00, 1.0],  # 0 = background
    [0.72, 0.90, 0.45, 1.0],  # 1 = area hasil merge/fill
    [0.18, 0.65, 0.22, 1.0],  # 2 = core kandidat awal
], dtype=float)

BORDER_COLOR = [1, 1, 1, 1]

# Nodata output.
# 255 dipakai agar nilai 0 tetap dapat bermakna background valid.
MASK_NODATA = 255
ZONE_NODATA = 0
CLASS_NODATA = 255

# Struktur ketetanggaan 8 arah.
# Artinya piksel horizontal, vertikal, dan diagonal dianggap bertetangga.
STRUCTURE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)


# =========================================================
# FUNGSI BANTU
# =========================================================
def build_outline(mask, thickness):
    """
    Membentuk garis tepi pada area final.
    Hanya digunakan jika BORDER_THICKNESS > 0.
    """
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
    Mengambil informasi depth dari nama file input.

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
    Menentukan file depth yang digunakan berdasarkan nama file input.
    """
    depth_suffix = get_depth_suffix(input_tif)

    depth_map = {
        "35m": ROOT_DEPTH / "Depth_35m_2025.tif",
        "10m": ROOT_DEPTH / "Depth_10m_2025.tif",
    }

    if depth_suffix not in depth_map:
        raise FileNotFoundError(
            f"Tidak ada mapping depth untuk suffix '{depth_suffix}' pada file {input_tif.name}"
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
    """
    Mengambil tahun dari nama file.
    """
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else None


def standardize_xy(da):
    """
    Menyeragamkan nama dimensi spasial menjadi x dan y.
    """
    rename_dict = {}

    if "longitude" in da.dims:
        rename_dict["longitude"] = "x"
    if "latitude" in da.dims:
        rename_dict["latitude"] = "y"

    da = da.rename(rename_dict)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da

def summarize_values(values, prefix):
    """
    Menghitung statistik min, max, mean, dan median untuk setiap zona.
    """
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
    Membentuk mask piksel yang memenuhi 4 faktor:
    salinitas, kedalaman, temperatur, dan sedimentasi.

    Mask ini digunakan untuk skenario check4factor.
    Pada skenario tersebut, piksel hasil merge/fill hanya dipertahankan
    jika memenuhi seluruh batas faktor lingkungan.
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
    Memuat salinitas, temperatur, sedimentasi, dan kedalaman,
    lalu menyamakan grid-nya dengan raster input.

    Semua faktor lingkungan harus berada pada grid yang sama
    agar pengecekan piksel dilakukan pada posisi spasial yang sepadan.
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

    # rata-rata tahunan
    cur_sal = sal.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
    cur_sal = cur_sal.sel(depth=0, method="nearest")
    
    cur_temp = temp.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
    cur_temp = cur_temp.sel(depth=0, method="nearest")
    
    cur_sen = sen.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")

    cur_sal = standardize_xy(cur_sal)
    cur_temp = standardize_xy(cur_temp)
    cur_sen = standardize_xy(cur_sen)

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
    common = f"{MERGE_GAP_PIXELS}_{base_name}_{scenario_name}"

    return {
        "zone_tif": output_dir / f"{common}_zone.tif",
        "png": output_dir / f"{common}_peta.png",
        "csv": output_dir / f"{common}_ringkasan.csv",
    }


def process_scenario(
    input_tif,
    arr,
    profile,
    valid_mask,
    candidate_mask,
    search_mask,
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
    Memproses satu skenario penggabungan.

    Skenario True / check4factor:
    - Area pencarian berasal dari hasil merge/fill kandidat awal.
    - Area akhir hanya diambil dari piksel dalam search_mask
      yang memenuhi 4 faktor lingkungan.
    - Rumus:
        final search = search_mask ∩ factor4_mask

    Skenario False / noCheck4factor:
    - Area pencarian berasal dari hasil merge/fill kandidat awal.
    - Seluruh area hasil merge/fill dipertahankan tanpa cek ulang 4 faktor.
    - Rumus:
        final search = search_mask

    Catatan:
    Logika ini sengaja mengikuti tujuan penelitian:
    piksel yang bukan hasil pemetaan awal tetap boleh menjadi area potensial
    apabila berada di sekitar kandidat awal dan memenuhi 4 faktor lingkungan.
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

    if check_merged_pixels_with_4_factors:
        # Skenario utama sesuai tujuan:
        # mencari piksel lain di sekitar kandidat awal yang memenuhi 4 faktor.
        working_mask = search_mask & factor4_mask
    else:
        # Skenario pembanding:
        # semua piksel hasil merge/fill dipertahankan tanpa cek ulang 4 faktor.
        working_mask = search_mask.copy()

    # =====================================================
    # LABELING KOMPONEN TERHUBUNG
    # =====================================================
    # Setiap kumpulan piksel yang saling terhubung diberi label zona.
    labeled, num_features = ndimage.label(working_mask, structure=STRUCTURE)
    component_sizes = np.bincount(labeled.ravel())

    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    zone_arr = np.zeros_like(labeled, dtype=np.int32)

    zone_stats = []
    new_id = 1

    # =====================================================
    # FILTER LUAS MINIMUM
    # =====================================================
    # Zona yang luasnya di bawah MIN_AREA tidak dipertahankan.
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

    # =====================================================
    # SIMPAN CSV RINGKASAN ZONA
    # =====================================================
    print("Ringkasan zona akhir:")
    if len(zone_stats) == 0:
        print("Tidak ada zona yang lolos luas minimum.")
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
    # =====================================================
    # 255 = nodata
    # 0   = background valid
    # 1   = area final hasil merge/fill
    # 2   = core area, yaitu piksel kandidat awal bernilai 4
    class_arr = np.full(arr.shape, CLASS_NODATA, dtype=np.uint8)
    class_arr[valid_mask] = 0
    class_arr[final_mask] = 1

    highest_value = max(POTENTIAL_VALUES)
    core_mask = valid_mask & (arr == highest_value) & final_mask
    class_arr[core_mask] = 2

    print("Jumlah piksel final:", int(np.count_nonzero(final_mask)))
    print("Jumlah piksel core :", int(np.count_nonzero(core_mask)))
    print(
        "Jumlah piksel ekstensi:",
        int(np.count_nonzero(final_mask & ~candidate_mask))
    )

    # =====================================================
    # SIMPAN ZONE RASTER
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
    """
    Memproses satu raster factor_suitability.

    Alur utama:
    1. Membaca raster factor_suitability.
    2. Membentuk candidate_mask dari piksel bernilai 4.
    3. Memuat 4 faktor lingkungan.
    4. Membentuk factor4_mask.
    5. Membentuk search_mask melalui merge dan fill holes.
    6. Memproses dua skenario:
       - check4factor
       - noCheck4factor
    """
    depth_suffix = get_depth_suffix(input_tif)
    output_dir = ROOT / depth_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_name(input_tif.stem)

    print("\n====================================================")
    print(f"Memproses : {input_tif.name}")
    print(f"Folder out: {output_dir}")
    print("====================================================")

    # =====================================================
    # 1. BACA RASTER INPUT
    # =====================================================
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        input_nodata = src.nodata

    print("Shape:", arr.shape)
    print("CRS:", profile.get("crs"))

    if input_nodata is not None:
        valid_mask = arr != input_nodata
    else:
        valid_mask = np.isfinite(arr)

    # =====================================================
    # 2. BENTUK KANDIDAT AWAL
    # =====================================================
    # Kandidat awal adalah piksel hasil pemetaan/factor_suitability
    # yang bernilai 4.
    candidate_mask = valid_mask & np.isin(arr, POTENTIAL_VALUES)

    print("Jumlah piksel valid:", int(np.count_nonzero(valid_mask)))
    print("Jumlah piksel kandidat awal:", int(np.count_nonzero(candidate_mask)))

    # =====================================================
    # 3. LOAD 4 FAKTOR LINGKUNGAN
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

    # =====================================================
    # 4. BENTUK FACTOR4 MASK
    # =====================================================
    # factor4_mask menunjukkan piksel yang memenuhi salinitas,
    # kedalaman, temperatur, dan sedimentasi.
    factor4_mask = build_factor4_mask(
        sal_arr=sal_arr,
        depth_arr=depth_arr,
        temp_arr=temp_arr,
        sen_arr=sen_arr
    )

    print("Jumlah piksel yang lolos 4 faktor:", int(np.count_nonzero(factor4_mask)))
    print(
        "Jumlah kandidat awal yang juga lolos 4 faktor:",
        int(np.count_nonzero(candidate_mask & factor4_mask))
    )

    # =====================================================
    # 5. BENTUK AREA PENCARIAN DARI KANDIDAT AWAL
    # =====================================================
    # search_mask adalah area sekitar kandidat awal.
    # Area ini dibentuk dengan binary closing dan fill holes.
    # Tujuannya bukan hanya menyatukan kandidat awal,
    # tetapi membuka ruang pencarian piksel tambahan di sekitarnya.
    search_mask = candidate_mask.copy()

    if MERGE_GAP_PIXELS > 0:
        search_mask = ndimage.binary_closing(
            search_mask,
            structure=STRUCTURE,
            iterations=MERGE_GAP_PIXELS
        )

    if FILL_HOLES:
        search_mask = ndimage.binary_fill_holes(search_mask)

    print(
        "Jumlah piksel tambahan yang lolos 4 faktor:",
        int(np.count_nonzero((search_mask & ~candidate_mask) & factor4_mask))
    )

    # =====================================================
    # 6. PROSES DUA SKENARIO
    # =====================================================
    for check_merged in CHECK_MERGED_SCENARIOS:
        process_scenario(
            input_tif=input_tif,
            arr=arr,
            profile=profile,
            valid_mask=valid_mask,
            candidate_mask=candidate_mask,
            search_mask=search_mask,
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