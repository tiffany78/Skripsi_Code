#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:42:26 2026

@author: tipanoii
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
import rioxarray
import re

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = Path("D:/TA/code/depth/")
TIF_PATTERN = "*.tif"
OUT_DIR = ROOT 

# Jika ingin cari sampai ke subfolder, gunakan rglob
tif_files = sorted(ROOT.rglob(TIF_PATTERN))

# =========================================================
# PENGAMBILAN HANYA PIKSEL YANG MERUPAKAN PERAIRAN
# =========================================================
def get_valid_values(depth_arr, nodata_value=None):
    """
    Mengambil nilai valid dari array kedalaman.
    """
    if hasattr(depth_arr, "values"):
        arr = depth_arr.values
    else:
        arr = np.array(depth_arr)

    arr = arr.astype(float)
    valid_mask = np.isfinite(arr)

    if nodata_value is not None:
        valid_mask &= (arr != nodata_value)

    return arr, valid_mask, arr[valid_mask]

# =========================================================
# STATISTIK DESKRIPTIF
# =========================================================
def print_descriptive_statistics(depth_arr, title):
    """
    Mencetak statistik deskriptif kedalaman.
    Statistik yang dihitung: minimum, maksimum, rata-rata, median, dan standar deviasi.
    """
    _, _, valid_vals = get_valid_values(depth_arr)

    if valid_vals.size == 0:
        print("Tidak ada nilai valid untuk statistik deskriptif.")
        return

    print("Statistik deskriptif kedalaman")
    print(f"Data               : {title}")
    print(f"Jumlah piksel valid: {valid_vals.size:,}")
    print(f"Minimum            : {np.min(valid_vals):.3f} m")
    print(f"Maksimum           : {np.max(valid_vals):.3f} m")
    print(f"Rata-rata          : {np.mean(valid_vals):.3f} m")
    print(f"Median             : {np.median(valid_vals):.3f} m")
    print(f"Standar deviasi    : {np.std(valid_vals):.3f} m")
    
# =========================================================
# ANALISIS KEDALAMAN PER KELAS
# =========================================================

# Interval dan kelas kedalaman berdasarkan kedalaman maksimum.
# File 10 m menggunakan interval 2 m: 0-2, 2-4, 4-6, 6-8, 8-10.
# File 35 m menggunakan interval 5 m: 0-5, 5-10, ..., 30-35.
DEPTH_ANALYSIS_CONFIG = {
    10: {
        "bin_width": 2,
        "class_edges": [0, 2, 4, 6, 8, 10],
    },
    35: {
        "bin_width": 5,
        "class_edges": [0, 5, 10, 15, 20, 25, 30, 35],
    },
}

# Konfigurasi cadangan jika nama file tidak mengandung informasi 10m atau 35m.
DEFAULT_DEPTH_BIN_WIDTH = 1
DEFAULT_DEPTH_CLASS_EDGES = [0, 2, 5, 10, 20, 35]


def extract_max_depth_m(path):
    """
    Mengambil informasi kedalaman maksimum dari nama file.

    Contoh nama file yang dapat dibaca:
    - depth_2023_10m.tif
    - Depth_Batnas_35m_2025.tif
    - hasil-2024-10 m.tif

    Fungsi mengembalikan angka 10 atau 35 jika ditemukan.
    Jika tidak ditemukan, fungsi mengembalikan None.
    """
    stem = path.stem.lower()
    match = re.search(r"(?:^|[_\-\s])(10|35)\s*m(?:$|[_\-\s])", stem)

    if match is None:
        return None

    return int(match.group(1))

def get_depth_analysis_config(path):
    """
    Menentukan interval visualisasi dan kelas luas berdasarkan kedalaman maksimum pada nama file.
    """
    max_depth = extract_max_depth_m(path)

    if max_depth in DEPTH_ANALYSIS_CONFIG:
        config = DEPTH_ANALYSIS_CONFIG[max_depth]
        return max_depth, config["bin_width"], config["class_edges"]

    print("Informasi kedalaman maksimum 10m atau 35m tidak ditemukan pada nama file.")
    print("Kode menggunakan konfigurasi default.")
    return None, DEFAULT_DEPTH_BIN_WIDTH, DEFAULT_DEPTH_CLASS_EDGES

# =========================================================
# PERHITUNGAN LUAS PER KELAS KEDALAMAN
# =========================================================
def get_pixel_area_m2(depth_arr):
    """
    Menghitung luas piksel dalam meter persegi.
    CRS projected, luas piksel dihitung dari resolusi raster.
    """
    crs = depth_arr.rio.crs

    # Jika CRS tidak ada, gunakan fallback ukuran piksel jika disediakan.
    if crs is None:
        print("Tidak ada keterangan CRS")
        return 

    res_x, res_y = depth_arr.rio.resolution()

    # Jika satuan CRS sudah meter, luas piksel konstan.
    if crs.is_projected:
        return abs(res_x * res_y)

def print_area_by_depth_class(depth_arr, title, class_edges=None):
    """
    Mencetak jumlah piksel, luas area dalam hektare, dan persentase luas
    berdasarkan jumlah piksel valid pada tahun tersebut.
    """
    if class_edges is None:
        class_edges = DEFAULT_DEPTH_CLASS_EDGES

    arr, valid_mask, valid_vals = get_valid_values(depth_arr)

    if valid_vals.size == 0:
        print("Tidak ada nilai valid untuk perhitungan luas.")
        return

    # Total piksel valid pada raster/tahun tersebut
    total_valid_pixels = int(np.sum(valid_mask))

    pixel_area_m2 = get_pixel_area_m2(depth_arr)

    if pixel_area_m2 is None:
        print("Luas per kelas kedalaman tidak dapat dihitung karena CRS tidak tersedia.")
        print("Isi FALLBACK_PIXEL_SIZE_M, misalnya 20, jika ukuran piksel diketahui.")
        return

    # Jika terdapat kedalaman yang lebih besar dari batas terakhir,
    # tambahkan kelas tambahan.
    edges = list(class_edges)
    max_depth = float(np.max(valid_vals))

    if max_depth > edges[-1]:
        new_max_edge = np.ceil(max_depth)

        if new_max_edge <= edges[-1]:
            new_max_edge = edges[-1] + 1

        edges.append(new_max_edge)

    print("Luas area per kelas kedalaman")
    print(f"Data: {title}")
    print(f"Total piksel valid: {total_valid_pixels:,}")
    print("Kelas kedalaman | Jumlah piksel | Luas (ha) | Persentase luas (%)")

    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]

        # Batas atas pada kelas terakhir dibuat inklusif.
        if i == len(edges) - 2:
            class_mask = valid_mask & (arr >= lower) & (arr <= upper)
        else:
            class_mask = valid_mask & (arr >= lower) & (arr < upper)

        pixel_count = int(np.sum(class_mask))

        # Hitung luas area
        if np.isscalar(pixel_area_m2):
            area_m2 = pixel_count * pixel_area_m2
        else:
            area_m2 = float(np.sum(np.where(class_mask, pixel_area_m2, 0)))

        area_ha = area_m2 / 10_000

        # Persentase berdasarkan jumlah piksel valid pada tahun tersebut
        percentage = (pixel_count / total_valid_pixels) * 100

        print(
            f"{lower:>5.1f} - {upper:<5.1f} m | "
            f"{pixel_count:>12,} | "
            f"{area_ha:>9.4f} | "
            f"{percentage:>18.2f}"
        )

# =========================================================
# EXPORT VISUALISASI DEPTH
# =========================================================
def save_depth_png(depth_arr, out_png, title="Depth Map", nodata_value=None, bin_width=0.5, show_plot=True):
    """
    Simpan depth raster menjadi PNG dengan legenda interval depth.
    ----------
    depth_arr : numpy array / xarray DataArray
        Array depth 2D.
    out_png : str or Path
        Path output PNG.
    title : str
        Judul peta.
    nodata_value : float/int or None
        Nilai nodata jika ada.
    bin_width : float
        Lebar interval depth.
    show_plot : bool
        Jika True, tampilkan plot.
    """
    if hasattr(depth_arr, "values"):
        arr = depth_arr.values
    else:
        arr = np.array(depth_arr)

    arr = arr.astype(float)

    # Mask valid
    valid_mask = np.isfinite(arr)
    if nodata_value is not None:
        valid_mask &= (arr != nodata_value)

    valid_vals = arr[valid_mask]

    if valid_vals.size == 0:
        raise ValueError("Tidak ada nilai depth valid untuk divisualisasikan.")

    # Batas interval keseluruhan
    dmin = np.floor(valid_vals.min() / bin_width) * bin_width
    dmax = np.ceil(valid_vals.max() / bin_width) * bin_width
    edges = np.arange(dmin, dmax + bin_width, bin_width)

    # Jaga kalau cuma ada satu nilai
    if len(edges) < 2:
        edges = np.array([dmin, dmin + bin_width])

    labels = [f"{edges[i]:.1f} - {edges[i+1]:.1f} m" for i in range(len(edges) - 1)]

    # Array hasil klasifikasi warna
    class_arr = np.full(arr.shape, -1, dtype=int)

    arr_adj = arr.copy()
    arr_adj[arr_adj == dmax] = np.nextafter(dmax, -np.inf)

    for i in range(len(edges) - 1):
        mask_bin = valid_mask & (arr_adj >= edges[i]) & (arr_adj < edges[i+1])
        class_arr[mask_bin] = i

    # Buat RGB
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    # nodata / invalid = hitam
    rgb[:, :] = [0, 0, 0]

    # Gunakan colormap bertingkat
    cmap = colormaps["Blues"]
    n_classes = len(labels)

    legend_elements = []

    for i in range(n_classes):
        color = cmap(i / max(n_classes - 1, 1))[:3]
        color_255 = tuple(int(c * 255) for c in color)
        rgb[class_arr == i] = color_255

        legend_elements.append(
            Patch(
                facecolor=np.array(color),
                edgecolor="black",
                label=labels[i]
            )
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    ax.legend(
        handles=legend_elements,
        title="Depth interval",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"PNG depth tersimpan: {out_png}")

# =========================================================
# PROSES SEMUA FILE TIFF
# =========================================================
if not tif_files:
    print(f"Tidak ada file TIFF yang ditemukan di: {ROOT}")
else:
    print(f"Jumlah file TIFF ditemukan: {len(tif_files)}")
    print("====================================")

    for depth_tif in tif_files:
        try:
            print(f"Memproses: {depth_tif.name}")

            # Baca GeoTIFF
            depth = rioxarray.open_rasterio(depth_tif).squeeze()

            print("Shape depth original:", depth.shape)
            print("CRS depth original:", depth.rio.crs)

            # Hapus nodata bawaan jika ada
            if depth.rio.nodata is not None:
                depth = depth.where(depth != depth.rio.nodata)

            # Ubah menjadi positif jika raster bathymetry negatif
            depth = np.abs(depth)

            # Judul PNG dari nama file GeoTIFF
            title_png = depth_tif.stem.replace("_", " ").replace("-", " ")

            # Nama output PNG disamakan dengan nama TIFF
            out_depth_png = OUT_DIR / f"{depth_tif.stem}.png"
            
            # Tentukan interval berdasarkan nama file.
            # File 10m menggunakan interval 2 m, sedangkan file 35m menggunakan interval 5 m.
            max_depth_m, depth_bin_width, depth_class_edges = get_depth_analysis_config(depth_tif)

            if max_depth_m is not None:
                print(f"Kedalaman maksimum terdeteksi: {max_depth_m} m")
                print(f"Interval visualisasi dan kelas luas: {depth_bin_width} m")
                
            # Cetak statistik deskriptif.
            print_descriptive_statistics(depth, title_png)

            # Cetak luas area per kelas kedalaman sesuai kedalaman maksimum.
            print_area_by_depth_class(depth, title_png, class_edges=depth_class_edges)

            # Simpan PNG
            save_depth_png(
                depth_arr=depth,
                out_png=out_depth_png,
                title=title_png,
                nodata_value=None,
                bin_width=1,
                show_plot=True
            )

            print("Selesai.")
            print("------------------------------------")

        except Exception as e:
            print(f"Gagal memproses {depth_tif.name}: {e}")
            print("------------------------------------")