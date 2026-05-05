#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:15:45 2026

@author: tipanoii
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = ""
MODE = "reef" # "seaweed" or "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS RUMPUT LAUT ========================")
    ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed")
elif MODE == "reef":
    print("\n============== ANALISIS TERUMBU KARANG ========================")
    ROOT = Path("/Users/tipanoii/doc/TA/code/reef")
else:
    raise SystemExit("Tipe Analisis Tidak Sesuai")
    
INPUT_FOLDER = ROOT / "input"
OUTPUT_TIF_FOLDER = ROOT / "output_frequency_tif"
OUTPUT_PNG_FOLDER = ROOT / "output_frequency_png"
SHOW_PLOT = False

TIF_PATTERN = "*.tif"
NODATA_VALUE = -999
EXPECTED_QUARTERS = ["q1", "q2", "q3", "q4"]

OUTPUT_TIF_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNGSI PARSE NAMA FILE
# Contoh: 2023_q1_B8A_S20_Laut.tif -> year=2023, quarter=q1
# =========================================================
def parse_year_quarter(tif_path):
    parts = tif_path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Format nama file tidak sesuai: {tif_path.name}")
    year = parts[0]
    quarter = parts[1].lower()
    return year, quarter

# =========================================================
# KELOMPOKKAN FILE BERDASARKAN TAHUN
# =========================================================
def group_files_by_year(input_folder):
    grouped = defaultdict(dict)

    for tif_path in sorted(Path(input_folder).glob(TIF_PATTERN)):
        year, quarter = parse_year_quarter(tif_path)

        if quarter in EXPECTED_QUARTERS:
            grouped[year][quarter] = tif_path
        else:
            print(f"File dilewati karena quarter tidak dikenali: {tif_path.name}")

    return grouped

# =========================================================
# VALIDASI UKURAN / CRS / TRANSFORM
# =========================================================
def validate_rasters(raster_dict):
    reference_shape = None
    reference_transform = None
    reference_crs = None

    for quarter, tif_path in raster_dict.items():
        with rasterio.open(tif_path) as src:
            shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs

            if reference_shape is None:
                reference_shape = shape
                reference_transform = transform
                reference_crs = crs
            else:
                if shape != reference_shape:
                    raise ValueError(
                        f"Ukuran raster tidak sama. File: {tif_path.name}, "
                        f"shape={shape}, seharusnya={reference_shape}"
                    )
                if transform != reference_transform:
                    raise ValueError(
                        f"Transform raster tidak sama. File: {tif_path.name}"
                    )
                if crs != reference_crs:
                    raise ValueError(
                        f"CRS raster tidak sama. File: {tif_path.name}"
                    )

def validate_rastersOld(raster_dict):
    reference_shape = None
    reference_transform = None
    reference_crs = None

    for q in EXPECTED_QUARTERS:
        tif_path = raster_dict[q]
        with rasterio.open(tif_path) as src:
            shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs

            if reference_shape is None:
                reference_shape = shape
                reference_transform = transform
                reference_crs = crs
            else:
                if shape != reference_shape:
                    raise ValueError(
                        f"Ukuran raster tidak sama. File: {tif_path.name}, "
                        f"shape={shape}, seharusnya={reference_shape}"
                    )
                if transform != reference_transform:
                    raise ValueError(
                        f"Transform raster tidak sama. File: {tif_path.name}"
                    )
                if crs != reference_crs:
                    raise ValueError(
                        f"CRS raster tidak sama. File: {tif_path.name}"
                    )

# =========================================================
# HITUNG FREKUENSI TAHUNAN
# Aturan:
# - jika suatu piksel pada salah satu quarter = -999,
#   maka piksel itu dikeluarkan dari seluruh tahun
# - frekuensi dihitung dari jumlah quarter yang bernilai 1
# =========================================================
def create_yearly_frequency(raster_dict):
    validate_rasters(raster_dict)

    arrays = []
    profile = None

    for quarter, tif_path in sorted(raster_dict.items()):
        with rasterio.open(tif_path) as src:
            arr = src.read(1)

            if profile is None:
                profile = src.profile.copy()

            arrays.append(arr)

    stack = np.stack(arrays, axis=0)

    # jika suatu piksel pada salah satu quarter yang tersedia = -999,
    # maka piksel itu dikeluarkan
    invalid_mask = np.any(stack == NODATA_VALUE, axis=0)

    # hitung frekuensi kemunculan kelas 1
    frequency = np.sum(stack == 1, axis=0).astype(np.int16)

    frequency[invalid_mask] = NODATA_VALUE

    profile.update(
        dtype=rasterio.int16,
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw"
    )

    return frequency, profile

def create_yearly_frequencyOld(raster_dict):
    validate_rasters(raster_dict)

    arrays = []
    profile = None

    for q in EXPECTED_QUARTERS:
        tif_path = raster_dict[q]
        with rasterio.open(tif_path) as src:
            arr = src.read(1)

            if profile is None:
                profile = src.profile.copy()

            arrays.append(arr)

    stack = np.stack(arrays, axis=0)  # shape = (4, rows, cols)

    # Mask tahunan:
    # bila ada satu quarter saja bernilai -999, piksel dikeluarkan total
    invalid_mask = np.any(stack == NODATA_VALUE, axis=0)

    # Hitung frekuensi kemunculan kelas 1
    frequency = np.sum(stack == 1, axis=0).astype(np.int16)

    # Piksel invalid diberi -999
    frequency[invalid_mask] = NODATA_VALUE

    # Update profile untuk output
    profile.update(
        dtype=rasterio.int16,
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw"
    )

    return frequency, profile

# =========================================================
# SIMPAN GEOTIFF FREKUENSI
# =========================================================
def save_frequency_geotiff(freq_arr, profile, out_tif):
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(freq_arr, 1)
    print(f"GeoTIFF frekuensi tersimpan: {out_tif}")

# =========================================================
# SIMPAN PNG FREKUENSI
# -999 = hitam
# 0 = putih
# 1 = kuning
# 2 = oranye
# 3 = merah
# 4 = merah tua
# =========================================================
def save_frequency_png(freq_arr, out_png, title, show_plot=False):
    h, w = freq_arr.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Default hitam untuk -999 / area yang dikeluarkan
    rgb[:, :] = [0, 0, 0]

    # Frekuensi valid 0 - 4
    rgb[freq_arr == 0] = [255, 255, 255]   # putih
    rgb[freq_arr == 1] = [255, 255, 153]   # kuning muda
    rgb[freq_arr == 2] = [255, 204, 102]   # oranye muda
    rgb[freq_arr == 3] = [255, 102, 102]   # merah muda
    rgb[freq_arr == 4] = [204, 0, 0]       # merah tua

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array([255, 255, 255]) / 255.0, edgecolor="black", label="0"),
        Patch(facecolor=np.array([255, 255, 153]) / 255.0, edgecolor="black", label="1"),
        Patch(facecolor=np.array([255, 204, 102]) / 255.0, edgecolor="black", label="2"),
        Patch(facecolor=np.array([255, 102, 102]) / 255.0, edgecolor="black", label="3"),
        Patch(facecolor=np.array([204, 0, 0]) / 255.0, edgecolor="black", label="4"),
    ]

    ax.legend(
        handles=legend_elements,
        title="Frekuensi",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG frekuensi tersimpan: {out_png}")

# =========================================================
# PROSES SEMUA TAHUN
# =========================================================
def print_frequency_pixel_counts(year, freq_arr):
    print(f"\nRingkasan jumlah piksel frekuensi tahun {year}:")
    for freq in [1, 2, 3, 4]:
        count = int(np.count_nonzero(freq_arr == freq))
        print(f"  Frekuensi {freq}: {count} piksel")
        
def process_all_years():
    grouped_files = group_files_by_year(INPUT_FOLDER)

    if not grouped_files:
        print("Tidak ada file TIFF yang ditemukan.")
        return

    for year in sorted(grouped_files.keys()):
        quarter_files = grouped_files[year]

        if not quarter_files:
            print(f"Tahun {year} tidak memiliki file yang valid.")
            continue

        available_quarters = sorted(quarter_files.keys())

        print(f"\nMemproses tahun {year} ...")
        print(f"Quarter tersedia: {available_quarters}")

        for q in available_quarters:
            print(f"  {q}: {quarter_files[q].name}")

        freq_arr, profile = create_yearly_frequency(quarter_files)
        print_frequency_pixel_counts(year, freq_arr)

        quarter_label = "_".join(available_quarters)

        out_tif = OUTPUT_TIF_FOLDER / f"{year}_frequency_{quarter_label}.tif"
        out_png = OUTPUT_PNG_FOLDER / f"{year}_frequency_{quarter_label}.png"

        save_frequency_geotiff(freq_arr, profile, out_tif)
        save_frequency_png(
            freq_arr=freq_arr,
            out_png=out_png,
            title=f"Frekuensi Kemunculan Rumput Laut {year} ({', '.join(available_quarters)})",
            show_plot=SHOW_PLOT
        )
        
def process_all_yearsOld():
    grouped_files = group_files_by_year(INPUT_FOLDER)

    if not grouped_files:
        print("Tidak ada file TIFF yang ditemukan.")
        return

    for year in sorted(grouped_files.keys()):
        quarter_files = grouped_files[year]

        missing_quarters = [q for q in EXPECTED_QUARTERS if q not in quarter_files]
        if missing_quarters:
            print(
                f"Tahun {year} dilewati karena quarter belum lengkap. "
                f"Missing: {missing_quarters}"
            )
            continue

        print(f"\nMemproses tahun {year} ...")
        for q in EXPECTED_QUARTERS:
            print(f"  {q}: {quarter_files[q].name}")

        freq_arr, profile = create_yearly_frequency(quarter_files)
        print_frequency_pixel_counts(year, freq_arr)

        out_tif = OUTPUT_TIF_FOLDER / f"{year}_frequency_q1_q4.tif"
        out_png = OUTPUT_PNG_FOLDER / f"{year}_frequency_q1_q4.png"

        save_frequency_geotiff(freq_arr, profile, out_tif)
        save_frequency_png(
            freq_arr=freq_arr,
            out_png=out_png,
            title=f"Frekuensi Kemunculan Rumput Laut {year} (Q1-Q4)",
            show_plot=SHOW_PLOT
        )

# =========================================================
# JALANKAN
# =========================================================
if __name__ == "__main__":
    process_all_years()