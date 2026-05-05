#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:40:36 2026

@author: tipanoii
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
import csv
import re

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = ""
MODE = "seaweed" # "seaweed" or "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS RUMPUT LAUT ========================")
    ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/output_factor2")
elif MODE == "reef":
    print("\n============== ANALISIS TERUMBU KARANG ========================")
    ROOT = Path("/Users/tipanoii/doc/TA/code/reef/output_factor2")
else:
    raise SystemExit("Tipe Analisis Tidak Sesuai")

# ambil hanya file factor_suitability di folder ROOT
INPUT_TIFS = sorted(ROOT.glob("factor_suitability_*.tif"))

# nilai faktor yang dianggap potensial
POTENTIAL_VALUES = [4]

# resolusi piksel
PIXEL_SIZE = 20  # meter
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

# =========================================================
# PARAMETER PENGGABUNGAN
# =========================================================
MIN_PIXELS = 25          # minimal ukuran area akhir (1ha = 25 piksel)
MERGE_GAP_PIXELS = 1     # gap 1 piksel = 20 meter masih bisa digabung
FILL_HOLES = True        # isi lubang kecil di dalam area
USE_MERGED_AREA_AS_FINAL = True
# True  -> hasil akhir memakai area merge/closing
# False -> hasil akhir hanya piksel kandidat asli

# visualisasi
SHOW_PLOT = False        # sebaiknya False kalau batch banyak file
FIGSIZE = (10, 8)
PNG_DPI = 300
BORDER_THICKNESS = -1

# warna peta
# 0 = background
# 1 = zona potensial final
# 2 = zona inti / nilai tertinggi (mis. faktor = 3)
CLASS_COLORS = np.array([
    [0.00, 0.00, 0.00, 1.0],  # 0 hitam
    [0.72, 0.90, 0.45, 1.0],  # 1 hijau muda
    [0.18, 0.65, 0.22, 1.0],  # 2 hijau lebih gelap
], dtype=float)

BORDER_COLOR = [1, 1, 1, 1]   # border putih

# nodata output
MASK_NODATA = 0
ZONE_NODATA = 0
CLASS_NODATA = 0

# struktur 8-neighbor
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
    Membuat outline/border dari area final.
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
    Ambil 3 karakter terakhir dari nama file (tanpa ekstensi).
    Contoh:
    factor_suitability_2025_Depth_Batnas_Focal_35m.tif -> 35m
    """
    stem = input_tif.stem
    if len(stem) < 3:
        raise ValueError(f"Nama file terlalu pendek untuk diambil 3 karakter terakhir: {input_tif.name}")
    return stem[-3:]


def safe_name(text: str) -> str:
    """
    Rapikan nama file agar aman dipakai.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def process_one_tif(input_tif: Path):
    # =========================================================
    # NAMA OUTPUT BERDASARKAN 3 CHAR TERAKHIR
    # =========================================================
    depth_suffix = get_depth_suffix(input_tif)   # mis. 35m / 10m
    output_dir = ROOT / depth_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_name(input_tif.stem)
    potential_str = "_".join(map(str, POTENTIAL_VALUES))

    OUTPUT_MASK_TIF  = output_dir / f"zona_potensial_mask_{base_name}_{potential_str}.tif"
    OUTPUT_ZONE_TIF  = output_dir / f"zona_potensial_zone_{base_name}_{potential_str}.tif"
    OUTPUT_CLASS_TIF = output_dir / f"zona_potensial_class_{base_name}_{potential_str}.tif"
    OUTPUT_PNG       = output_dir / f"zona_potensial_peta_{base_name}_{potential_str}.png"
    OUTPUT_CSV       = output_dir / f"zona_potensial_ringkasan_{base_name}_{potential_str}.csv"

    print("\n====================================================")
    print(f"Memproses : {input_tif.name}")
    print(f"Folder out: {output_dir}")
    print("====================================================")

    # =========================================================
    # BACA DATA
    # =========================================================
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        input_nodata = src.nodata

    print("Shape:", arr.shape)
    print("CRS:", profile.get("crs"))
    print("Input nodata:", input_nodata)

    # valid pixel
    if input_nodata is not None:
        valid_mask = arr != input_nodata
    else:
        valid_mask = np.isfinite(arr)

    # kandidat area potensial
    candidate_mask = valid_mask & np.isin(arr, POTENTIAL_VALUES)
    print("Jumlah piksel kandidat awal:", int(np.count_nonzero(candidate_mask)))

    # =========================================================
    # MERGE AREA YANG BERDEKATAN
    # =========================================================
    working_mask = candidate_mask.copy()

    if MERGE_GAP_PIXELS > 0:
        working_mask = ndimage.binary_closing(
            working_mask,
            structure=STRUCTURE,
            iterations=MERGE_GAP_PIXELS
        )

    if FILL_HOLES:
        working_mask = ndimage.binary_fill_holes(working_mask)

    print("Jumlah piksel setelah merge/fill:", int(np.count_nonzero(working_mask)))

    # =========================================================
    # CONNECTED COMPONENT
    # =========================================================
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
            "area_m2": area_m2,
            "area_ha": area_ha
        })

        new_id += 1

    print("\nRingkasan zona akhir:")
    if len(zone_stats) == 0:
        print("Tidak ada zona yang lolos.")
    else:
        print(f"Total Zona: {new_id - 1}")

        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["zone_id", "pixel_count", "area_m2", "area_ha"]
            )
            writer.writeheader()
            writer.writerows(zone_stats)

        print(f"Ringkasan CSV tersimpan: {OUTPUT_CSV}")

    # =========================================================
    # BUAT CLASS ARRAY UNTUK VISUALISASI
    # =========================================================
    class_arr = np.zeros_like(arr, dtype=np.uint8)
    class_arr[final_mask] = 1

    highest_value = max(POTENTIAL_VALUES)
    core_mask = valid_mask & (arr == highest_value) & final_mask
    class_arr[core_mask] = 2

    print("Jumlah piksel final:", int(np.count_nonzero(final_mask)))
    print("Jumlah piksel core :", int(np.count_nonzero(core_mask)))

    # =========================================================
    # SIMPAN MASK GEOTIFF (0/1)
    # =========================================================
    mask_out = final_mask.astype(np.uint8)

    mask_profile = profile.copy()
    mask_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=MASK_NODATA,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_MASK_TIF, "w", **mask_profile) as dst:
        dst.write(mask_out, 1)

    print(f"Mask final tersimpan: {OUTPUT_MASK_TIF}")

    # =========================================================
    # SIMPAN ZONE ID GEOTIFF
    # =========================================================
    zone_profile = profile.copy()
    zone_profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=ZONE_NODATA,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_ZONE_TIF, "w", **zone_profile) as dst:
        dst.write(zone_arr, 1)

    print(f"Zone ID tersimpan: {OUTPUT_ZONE_TIF}")

    # =========================================================
    # SIMPAN CLASS GEOTIFF
    # =========================================================
    class_profile = profile.copy()
    class_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=CLASS_NODATA,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_CLASS_TIF, "w", **class_profile) as dst:
        dst.write(class_arr, 1)

    print(f"Class raster tersimpan: {OUTPUT_CLASS_TIF}")

    # =========================================================
    # BUAT PNG PETA
    # =========================================================
    cmap = ListedColormap(CLASS_COLORS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(class_arr, cmap=cmap, interpolation="nearest")

    if BORDER_THICKNESS > 0:
        outline_mask = build_outline(final_mask, thickness=BORDER_THICKNESS)
        outline_rgba = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=float)
        outline_rgba[outline_mask] = BORDER_COLOR
        ax.imshow(outline_rgba, interpolation="nearest")

    ax.set_title(
        "Peta Zona Potensial Budidaya Rumput Laut\n"
        f"Input={input_tif.name} | "
        f"Potential={POTENTIAL_VALUES} | "
        f"Min area={MIN_PIXELS} px | "
        f"Merge gap={MERGE_GAP_PIXELS} px ({MERGE_GAP_PIXELS * PIXEL_SIZE} m) | "
        f"Fill holes={FILL_HOLES}"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=PNG_DPI, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG peta tersimpan: {OUTPUT_PNG}")


# =========================================================
# MAIN LOOP
# =========================================================
if not INPUT_TIFS:
    raise FileNotFoundError(f"Tidak ada file factor_suitability_*.tif di folder: {ROOT}")

print("Jumlah file input ditemukan:", len(INPUT_TIFS))
for tif in INPUT_TIFS:
    try:
        process_one_tif(tif)
    except Exception as e:
        print(f"Gagal memproses {tif.name}: {e}")