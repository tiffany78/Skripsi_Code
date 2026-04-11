#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:22:53 2026

@author: tipanoii
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/B8A_S20_year/")
INPUT_FOLDER = ROOT
OUTPUT_TIF = ROOT / "all_files_frequency.tif"
OUTPUT_PNG = ROOT / "all_files_frequency.png"
SHOW_PLOT = False

TIF_PATTERN = "*.tif"
NODATA_VALUE = -999

# =========================================================
# VALIDASI SEMUA RASTER
# =========================================================
def validate_rasters(tif_files):
    reference_shape = None
    reference_transform = None
    reference_crs = None

    for tif_path in tif_files:
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
                    raise ValueError(f"Ukuran raster tidak sama: {tif_path.name}")
                if transform != reference_transform:
                    raise ValueError(f"Transform raster tidak sama: {tif_path.name}")
                if crs != reference_crs:
                    raise ValueError(f"CRS raster tidak sama: {tif_path.name}")

# =========================================================
# HITUNG FREKUENSI DARI SELURUH FILE BINARY
# =========================================================
def create_frequency_from_all_binary_files(input_folder):
    tif_files = sorted(Path(input_folder).glob(TIF_PATTERN))

    if not tif_files:
        raise ValueError("Tidak ada file TIFF ditemukan.")

    validate_rasters(tif_files)

    arrays = []
    profile = None

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            arr = src.read(1)

            if profile is None:
                profile = src.profile.copy()

            arrays.append(arr)

    stack = np.stack(arrays, axis=0)

    # jika suatu piksel pernah -999 di salah satu file, keluarkan total
    invalid_mask = np.any(stack == NODATA_VALUE, axis=0)

    # hitung berapa kali bernilai 1
    frequency = np.sum(stack == 1, axis=0).astype(np.int16)

    frequency[invalid_mask] = NODATA_VALUE

    profile.update(
        dtype=rasterio.int16,
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw"
    )

    return frequency, profile, tif_files

# =========================================================
# SIMPAN GEOTIFF
# =========================================================
def save_geotiff(arr, profile, out_tif):
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)
    print(f"GeoTIFF tersimpan: {out_tif}")

# =========================================================
# SIMPAN PNG DINAMIS
# =========================================================
def save_frequency_png_dynamic(freq_arr, out_png, title, nodata_value=-999, show_plot=False):
    valid_vals = freq_arr[freq_arr != nodata_value]
    max_freq = int(valid_vals.max()) if valid_vals.size > 0 else 0

    # warna dasar hitam
    rgb = np.zeros((freq_arr.shape[0], freq_arr.shape[1], 3), dtype=np.uint8)

    # untuk visual sederhana: makin besar frekuensi makin merah
    for v in range(0, max_freq + 1):
        if max_freq == 0:
            intensity = 255
        else:
            intensity = int(255 - (v / max_freq) * 180)
        rgb[freq_arr == v] = [255, intensity, intensity]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend_elements = []
    for v in range(0, max_freq + 1):
        if max_freq == 0:
            intensity = 255
        else:
            intensity = int(255 - (v / max_freq) * 180)
        legend_elements.append(
            Patch(facecolor=np.array([255, intensity, intensity]) / 255.0,
                  edgecolor="black", label=str(v))
        )

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

    print(f"PNG tersimpan: {out_png}")

# =========================================================
# PRINT RINGKASAN
# =========================================================
def print_frequency_counts(freq_arr, nodata_value=-999):
    valid_vals = freq_arr[freq_arr != nodata_value]
    max_freq = int(valid_vals.max()) if valid_vals.size > 0 else 0

    print("\nRingkasan jumlah piksel:")
    print(f"  NoData (-999): {int(np.count_nonzero(freq_arr == nodata_value))} piksel")

    for v in range(0, max_freq + 1):
        print(f"  Frekuensi {v}: {int(np.count_nonzero(freq_arr == v))} piksel")

# =========================================================
# JALANKAN
# =========================================================
if __name__ == "__main__":
    freq_arr, profile, tif_files = create_frequency_from_all_binary_files(INPUT_FOLDER)

    print("File yang diproses:")
    for f in tif_files:
        print(" -", f.name)

    print_frequency_counts(freq_arr, NODATA_VALUE)
    save_geotiff(freq_arr, profile, OUTPUT_TIF)
    save_frequency_png_dynamic(
        freq_arr,
        OUTPUT_PNG,
        title=f"Frekuensi Seluruh File ({len(tif_files)} file)",
        nodata_value=NODATA_VALUE,
        show_plot=SHOW_PLOT
    )