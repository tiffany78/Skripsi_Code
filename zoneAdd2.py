#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:27:46 2026

@author: tipanoii
"""

from pathlib import Path
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
import csv

# =========================================================
# KONFIGURASI
# =========================================================
MODE = "reef"   # "seaweed" or "reef"

if MODE == "seaweed":
    ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/output_filtering")
    print("\n============== POST-PROCESS ZONA EKSTENSI RUMPUT LAUT ========================")
elif MODE == "reef":
    ROOT = Path("/Users/tipanoii/doc/TA/code/reef/output_filtering")
    print("\n============== POST-PROCESS ZONA EKSTENSI TERUMBU KARANG ========================")
else:
    raise SystemExit("MODE harus 'seaweed' atau 'reef'")

# input dari hasil ekstensi sebelumnya
INPUT_MASKS = sorted(ROOT.glob("*/*/zona_ekstensi_merged_mask_*.tif"))

# resolusi piksel
PIXEL_SIZE = 20  # meter
PIXEL_AREA_M2 = PIXEL_SIZE * PIXEL_SIZE

# parameter tahap kedua
MIN_AREA = 2  # ha
MIN_PIXELS = int((MIN_AREA * 10000) / PIXEL_AREA_M2)   # 2 ha
MERGE_GAP_PIXELS = 1   # 1 piksel = 20 meter
FILL_HOLES = True
USE_MERGED_AREA_AS_FINAL = True

# visualisasi
SHOW_PLOT = False
FIGSIZE = (10, 8)
PNG_DPI = 300
BORDER_THICKNESS = 1

# output nodata
MASK_NODATA = 0
ZONE_NODATA = 0

# warna peta
CLASS_COLORS = np.array([
    [0.00, 0.00, 0.00, 1.0],  # 0 background hitam
    [0.72, 0.90, 0.45, 1.0],  # 1 zona final hijau muda
], dtype=float)

BORDER_COLOR = [1, 1, 1, 1]  # putih

# struktur 8-neighbor
STRUCTURE = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)


# =========================================================
# FUNGSI BANTU
# =========================================================
def safe_name(text: str) -> str:
    prefix = "zona_ekstensi_merged_mask_factor_suitability_"
    if text.startswith(prefix):
        text = text[len(prefix):]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)

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


def process_one_mask(input_tif: Path):
    """
    Input:
        zona_ekstensi_merged_mask_*.tif  (0/1)

    Output:
        - zona_final_mask_*.tif
        - zona_final_zone_*.tif
        - zona_final_peta_*.png
        - zona_final_ringkasan_*.csv
    """
    output_dir = input_tif.parent.parent / "zona_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = safe_name(input_tif.stem)
    print(base_name)

    OUTPUT_MASK_TIF = output_dir / f"zona_final_mask_{base_name}.tif"
    OUTPUT_ZONE_TIF = output_dir / f"zona_final_zone_{base_name}.tif"
    OUTPUT_PNG      = output_dir / f"zona_final_peta_{base_name}.png"
    OUTPUT_CSV      = output_dir / f"zona_final_ringkasan_{base_name}.csv"

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

    print("Shape:", arr.shape)
    print("CRS:", profile.get("crs"))
    print("Input nodata:", input_nodata)

    if input_nodata is not None:
        valid_mask = arr != input_nodata
    else:
        valid_mask = np.isfinite(arr)

    # kandidat = piksel 1
    candidate_mask = valid_mask & (arr == 1)
    print("Jumlah piksel kandidat awal:", int(np.count_nonzero(candidate_mask)))

    # =====================================================
    # MERGE 20 METER
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

    print("Jumlah piksel setelah merge/fill:", int(np.count_nonzero(working_mask)))

    # =====================================================
    # CONNECTED COMPONENT
    # =====================================================
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

        # filter minimum 2 ha
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
            "area_ha": round(area_ha, 3)
        })

        new_id += 1

    print("\nRingkasan zona final tahap kedua:")
    if len(zone_stats) == 0:
        print("Tidak ada zona yang lolos.")
    else:
        print(f"Total zona final: {new_id - 1}")

        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["zone_id", "pixel_count", "area_m2", "area_ha"]
            )
            writer.writeheader()
            writer.writerows(zone_stats)

        print(f"CSV ringkasan tersimpan: {OUTPUT_CSV}")

    print("Jumlah piksel final:", int(np.count_nonzero(final_mask)))

    # =====================================================
    # SIMPAN MASK FINAL
    # =====================================================
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

    # =====================================================
    # SIMPAN ZONE ID FINAL
    # =====================================================
    zone_profile = profile.copy()
    zone_profile.update(
        dtype=rasterio.int32,
        count=1,
        nodata=ZONE_NODATA,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_ZONE_TIF, "w", **zone_profile) as dst:
        dst.write(zone_arr, 1)

    print(f"Zone ID final tersimpan: {OUTPUT_ZONE_TIF}")

    # =====================================================
    # PNG
    # =====================================================
    class_arr = np.zeros_like(arr, dtype=np.uint8)
    class_arr[final_mask] = 1

    cmap = ListedColormap(CLASS_COLORS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(class_arr, cmap=cmap, interpolation="nearest")

    if BORDER_THICKNESS > 0:
        outline_mask = build_outline(final_mask, thickness=BORDER_THICKNESS)
        outline_rgba = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=float)
        outline_rgba[outline_mask] = BORDER_COLOR
        ax.imshow(outline_rgba, interpolation="nearest")

    ax.set_title(
        "Post-Process Zona Ekstensi\n"
        f"Input={input_tif.name} | "
        f"Min area={MIN_AREA} ha ({MIN_PIXELS} px) | "
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
if not INPUT_MASKS:
    raise FileNotFoundError(
        f"Tidak ada file zona_ekstensi_merged_mask_*.tif di folder bertingkat: {ROOT}"
    )

print("Jumlah file input ditemukan:", len(INPUT_MASKS))
for tif in INPUT_MASKS:
    print(" -", tif)

for tif in INPUT_MASKS:
    try:
        process_one_mask(tif)
    except Exception as e:
        print(f"Gagal memproses {tif.name}: {e}")