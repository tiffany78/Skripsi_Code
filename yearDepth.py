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

# =========================================================
# KONFIGURASI
# =========================================================
ROOT = Path("/Users/tipanoii/doc/TA/code/depth/")
TIF_PATTERN = "*.tif"
OUT_DIR = ROOT 

# Jika ingin cari sampai ke subfolder, gunakan rglob
tif_files = sorted(ROOT.rglob(TIF_PATTERN))

# =========================================================
# EXPORT VISUALISASI DEPTH
# =========================================================
def save_depth_png(depth_arr, out_png, title="Depth Map", nodata_value=None, bin_width=0.5, show_plot=True):
    """
    Simpan depth raster menjadi PNG dengan legenda interval depth.

    Parameters
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
    cmap = colormaps["viridis"]
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