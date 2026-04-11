#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:56:38 2026

@author: tipanoii
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/B8A_S20_semester/")
INPUT_FOLDER = ROOT
PNG_FOLDER = ROOT / "output"
OUTPUT_CSV = ROOT / "ringkasan_luas_kelas.csv"
OUTPUT_GRAPH = ROOT / "grafik_perubahan_seaweed.png"
SHOW_PLOT = False

TIF_PATTERN = "*.tif"

# =========================================================
# FUNGSI HITUNG LUAS
# =========================================================
def calculate_class_areas(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)

        if src.crs is None or not src.crs.is_projected:
            raise ValueError(
                f"{tif_path.name}: CRS tidak projected. "
                "Kode ini mengasumsikan raster projected (mis. EPSG:32754) agar luas bisa dihitung langsung."
            )

        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)
        pixel_area_m2 = pixel_width * pixel_height

        count_0 = int(np.count_nonzero(arr == 0))
        count_1 = int(np.count_nonzero(arr == 1))

        area_0_m2 = count_0 * pixel_area_m2
        area_1_m2 = count_1 * pixel_area_m2
        total_area_m2 = area_0_m2 + area_1_m2
        percentage_seaweed = round((area_1_m2 / total_area_m2) * 100, 2)

        return {
            "file_name": "_".join(tif_path.stem.split("_")[:2]),
            "pixel_size_x": pixel_width,
            "pixel_size_y": pixel_height,
            "pixel_area_m2": pixel_area_m2,
            "total_area_ha": total_area_m2 / 10000.0,
            "area_class_0_ha": area_0_m2 / 10000.0,
            "area_class_1_ha": area_1_m2 / 10000.0,
            "percentage_seaweed": percentage_seaweed,
        }

# =========================================================
# FUNGSI BUAT PNG
# =========================================================
def save_png_from_geotiff(tif_path, output_folder, show_plot=SHOW_PLOT):
    tif_path = Path(tif_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        arr = src.read(1)

    # Default hitam untuk semua nilai selain 0 dan 1
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    # 0 = putih
    rgb[arr == 0] = [255, 255, 255]

    # 1 = merah
    rgb[arr == 1] = [255, 0, 0]

    # nilai lain tetap hitam

    out_png = output_folder / f"{tif_path.stem}.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(tif_path.stem)
    ax.axis("off")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG tersimpan: {out_png}")

# =========================================================
# PROSES SEMUA FILE
# =========================================================
PNG_FOLDER.mkdir(parents=True, exist_ok=True)

tif_files = sorted(INPUT_FOLDER.glob(TIF_PATTERN))

if len(tif_files) == 0:
    raise FileNotFoundError(f"Tidak ada file .tif di folder: {INPUT_FOLDER}")

results = []

for tif in tif_files:
    print(f"\nMemproses: {tif.name}")

    # buat PNG
    save_png_from_geotiff(tif, PNG_FOLDER, show_plot=SHOW_PLOT)

    # hitung luas
    info = calculate_class_areas(tif)
    results.append(info)

# =========================================================
# SIMPAN CSV
# =========================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nCSV ringkasan tersimpan: {OUTPUT_CSV}")
print(df[[
    "file_name",
    "area_class_1_ha", "percentage_seaweed"
]])

# =========================================================
# BUAT GRAFIK PERUBAHAN LUAS
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(df["file_name"], df["percentage_seaweed"], marker="o", linewidth=2)

for x, y in zip(df["file_name"], df["percentage_seaweed"]):
    plt.text(x, y, f"{y:.2f}%", ha="center", va="bottom", fontsize=9)

plt.title("Perubahan Persentase Area Potensial Rumput Laut")
plt.xlabel("Rentang Waktu")
plt.ylabel("Persentase Area Potensial (%)")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(OUTPUT_GRAPH, dpi=300, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close()

print(f"Grafik tersimpan: {OUTPUT_GRAPH}")