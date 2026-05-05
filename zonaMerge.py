#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:46:14 2026

@author: tipanoii
"""

from pathlib import Path
import re
import csv
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

# =========================================================
# KONFIGURASI
# =========================================================
MODE = "seaweed"   # "seaweed" or "reef"

if MODE == "seaweed":
    ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/output_filtering")
    print("\n============== ANALISIS MERGE POLYGON RUMPUT LAUT ========================")
elif MODE == "reef":
    ROOT = Path("/Users/tipanoii/doc/TA/code/reef/output_filtering")
    print("\n============== ANALISIS MERGE POLYGON TERUMBU KARANG ========================")
else:
    raise SystemExit("MODE harus 'seaweed' atau 'reef'")

# pilih salah satu sesuai input Anda
INPUT_MASKS = sorted(ROOT.glob("*/zona_potensial_mask_*.tif"))
# alternatif:
# INPUT_MASKS = sorted(ROOT.glob("*/zona_ekstensi_merged_mask_*.tif"))

PIXEL_SIZE = 20  # meter
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

MERGE_GAP_PIXELS = 3   # 1 piksel = 20 m
MIN_AREA_HA = 2
MIN_PIXELS = int((MIN_AREA_HA * 10000) / PIXEL_AREA_M2)

FILL_HOLES = True
USE_8_NEIGHBOR = True

SHOW_PLOT = False
PNG_DPI = 300
FIGSIZE = (10, 8)

MASK_NODATA = 0
ZONE_NODATA = 0

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


# =========================================================
# FUNGSI BANTU
# =========================================================
def safe_name(text: str) -> str:
    prefix = "zona_potensial_mask_factor_suitability_2025_Depth_Batnas_"
    if text.startswith(prefix):
        text = text[len(prefix):]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)

def build_outline(mask, structure, thickness=1):
    if thickness <= 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=1)
    edge = mask & (~eroded)

    if thickness > 1:
        edge = ndimage.binary_dilation(edge, structure=structure, iterations=thickness - 1)

    return edge

def summarize_group(original_ids, original_label_arr, merged_group_mask):
    """
    original_ids      : list polygon awal yang masuk grup
    original_label_arr: raster label polygon awal
    merged_group_mask : raster bool untuk grup hasil merge
    """
    # mask total polygon awal yang tergabung dalam grup ini
    original_union = np.isin(original_label_arr, original_ids)

    original_pixels = int(np.count_nonzero(original_union))
    merged_pixels   = int(np.count_nonzero(merged_group_mask))
    added_pixels    = merged_pixels - original_pixels

    return {
        "n_original_polygons": len(original_ids),
        "original_polygon_ids": ",".join(map(str, sorted(original_ids))),
        "original_pixel_count": original_pixels,
        "merged_pixel_count": merged_pixels,
        "added_gap_pixel_count": added_pixels,
        "original_area_ha": round(original_pixels * PIXEL_AREA_M2 / 10000.0, 3),
        "merged_area_ha": round(merged_pixels * PIXEL_AREA_M2 / 10000.0, 3),
        "added_gap_area_ha": round(added_pixels * PIXEL_AREA_M2 / 10000.0, 3),
    }


# =========================================================
# PROSES SATU FILE
# =========================================================
def process_one_mask(input_tif: Path):
    # hasil dimasukkan ke subfolder baru
    output_dir = input_tif.parent / "merge_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_name(input_tif.stem)

    OUT_ORIG_ZONE_TIF   = output_dir / f"merge_original_zone_{base_name}.tif"
    OUT_MERGED_MASK_TIF = output_dir / f"merge_mask_{base_name}.tif"
    OUT_MERGED_ZONE_TIF = output_dir / f"merge_zone_{base_name}.tif"
    OUT_FILTERED_MASK   = output_dir / f"merge_filtered_mask_{base_name}.tif"
    OUT_FILTERED_ZONE   = output_dir / f"merge_filtered_zone_{base_name}.tif"
    OUT_CSV             = output_dir / f"merge_analysis_{base_name}.csv"
    OUT_PNG             = output_dir / f"merge_analysis_{base_name}.png"

    print("\n====================================================")
    print(f"Memproses : {input_tif.name}")
    print(f"Output dir: {output_dir}")
    print("====================================================")

    # =====================================================
    # BACA MASK 0/1
    # =====================================================
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        input_nodata = src.nodata

    if input_nodata is not None:
        valid_mask = arr != input_nodata
    else:
        valid_mask = np.isfinite(arr)

    candidate_mask = valid_mask & (arr == 1)

    print("Jumlah piksel potensial awal:", int(np.count_nonzero(candidate_mask)))

    # =====================================================
    # LABEL POLYGON AWAL
    # =====================================================
    original_labels, original_num = ndimage.label(candidate_mask, structure=STRUCTURE)
    print("Jumlah polygon awal:", original_num)

    # =====================================================
    # MERGE DENGAN GAP METER
    # =====================================================
    working_mask = candidate_mask.copy()

    if MERGE_GAP_PIXELS > 0:
        working_mask = ndimage.binary_closing(
            working_mask,
            structure=STRUCTURE,
            iterations=MERGE_GAP_PIXELS
        )

    if FILL_HOLES:
        working_mask = ndimage.binary_fill_holes(working_mask)

    merged_labels, merged_num = ndimage.label(working_mask, structure=STRUCTURE)
    print("Jumlah grup setelah merge:", merged_num)

    # =====================================================
    # ANALISIS GRUP MERGE
    # =====================================================
    rows = []
    filtered_mask = np.zeros_like(candidate_mask, dtype=bool)
    filtered_zone = np.zeros_like(merged_labels, dtype=np.int32)

    new_id = 1

    for merged_id in range(1, merged_num + 1):
        merged_group_mask = merged_labels == merged_id

        # polygon awal apa saja yang masuk grup merge ini?
        orig_ids = np.unique(original_labels[merged_group_mask])
        orig_ids = [int(v) for v in orig_ids if v > 0]

        if len(orig_ids) == 0:
            continue

        stats = summarize_group(orig_ids, original_labels, merged_group_mask)

        row = {
            "merged_group_id": merged_id,
            "final_zone_id": None,   # akan diisi kalau lolos filter
            "merge_gap_pixels": MERGE_GAP_PIXELS,
            "merge_gap_m": MERGE_GAP_PIXELS * PIXEL_SIZE,
            "min_area_ha_threshold": MIN_AREA_HA,
            "passes_min_area": stats["merged_area_ha"] >= MIN_AREA_HA,
        }
        row.update(stats)

        if stats["merged_pixel_count"] >= MIN_PIXELS:
            filtered_mask |= merged_group_mask
            filtered_zone[merged_group_mask] = new_id
            row["final_zone_id"] = new_id
            new_id += 1

        rows.append(row)

    # =====================================================
    # SIMPAN CSV
    # =====================================================
    fieldnames = [
        "merged_group_id",
        "final_zone_id",
        "merge_gap_pixels",
        "merge_gap_m",
        "min_area_ha_threshold",
        "passes_min_area",
        "n_original_polygons",
        "original_polygon_ids",
        "original_pixel_count",
        "merged_pixel_count",
        "added_gap_pixel_count",
        "original_area_ha",
        "merged_area_ha",
        "added_gap_area_ha",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV analisis tersimpan: {OUT_CSV}")

    # =====================================================
    # SIMPAN RASTER
    # =====================================================
    # 1. original labels
    p1 = profile.copy()
    p1.update(dtype=rasterio.int32, count=1, nodata=ZONE_NODATA, compress="lzw")
    with rasterio.open(OUT_ORIG_ZONE_TIF, "w", **p1) as dst:
        dst.write(original_labels.astype(np.int32), 1)

    # 2. merged mask
    p2 = profile.copy()
    p2.update(dtype=rasterio.uint8, count=1, nodata=MASK_NODATA, compress="lzw")
    with rasterio.open(OUT_MERGED_MASK_TIF, "w", **p2) as dst:
        dst.write(working_mask.astype(np.uint8), 1)

    # 3. merged zone
    p3 = profile.copy()
    p3.update(dtype=rasterio.int32, count=1, nodata=ZONE_NODATA, compress="lzw")
    with rasterio.open(OUT_MERGED_ZONE_TIF, "w", **p3) as dst:
        dst.write(merged_labels.astype(np.int32), 1)

    # 4. filtered final mask (>= 2 ha)
    p4 = profile.copy()
    p4.update(dtype=rasterio.uint8, count=1, nodata=MASK_NODATA, compress="lzw")
    with rasterio.open(OUT_FILTERED_MASK, "w", **p4) as dst:
        dst.write(filtered_mask.astype(np.uint8), 1)

    # 5. filtered final zone
    p5 = profile.copy()
    p5.update(dtype=rasterio.int32, count=1, nodata=ZONE_NODATA, compress="lzw")
    with rasterio.open(OUT_FILTERED_ZONE, "w", **p5) as dst:
        dst.write(filtered_zone.astype(np.int32), 1)

    print(f"Raster hasil merge tersimpan di: {output_dir}")

    # =====================================================
    # PNG VISUAL
    # 0 = background
    # 1 = polygon awal
    # 2 = area tambahan karena merge
    # 3 = hasil final lolos filter
    # =====================================================
    class_arr = np.zeros_like(arr, dtype=np.uint8)
    class_arr[candidate_mask] = 1

    added_gap_mask = working_mask & (~candidate_mask)
    class_arr[added_gap_mask] = 2

    class_arr[filtered_mask] = 3

    # kembalikan polygon awal di atas final supaya masih terlihat
    class_arr[candidate_mask] = 1

    colors = np.array([
        [0.00, 0.00, 0.00, 1.0],  # 0 background hitam
        [1.00, 0.90, 0.00, 1.0],  # 1 polygon awal kuning
        [1.00, 0.20, 0.20, 1.0],  # 2 celah tambahan merah
        [0.35, 0.85, 0.85, 1.0],  # 3 hasil merge final cyan
    ], dtype=float)

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(class_arr, cmap=cmap, interpolation="nearest")

    outline_mask = build_outline(filtered_mask, STRUCTURE, thickness=0)
    outline_rgba = np.zeros((filtered_mask.shape[0], filtered_mask.shape[1], 4), dtype=float)
    outline_rgba[outline_mask] = [1, 1, 1, 1]
    ax.imshow(outline_rgba, interpolation="nearest")

    ax.set_title(
        f"Analisis Merge Polygon\n"
        f"{input_tif.name}\n"
        f"Gap={MERGE_GAP_PIXELS} px ({MERGE_GAP_PIXELS * PIXEL_SIZE} m) | "
        f"Min area={MIN_AREA_HA} ha"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG analisis tersimpan: {OUT_PNG}")


# =========================================================
# MAIN
# =========================================================
if not INPUT_MASKS:
    raise FileNotFoundError(f"Tidak ada file mask input di: {ROOT}")

print("Jumlah file input ditemukan:", len(INPUT_MASKS))
for f in INPUT_MASKS:
    print(" -", f)

for input_tif in INPUT_MASKS:
    try:
        process_one_mask(input_tif)
    except Exception as e:
        print(f"Gagal memproses {input_tif.name}: {e}")