#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:51:44 2026

@author: tipanoii
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================================================
# PATH
# =========================================================
ROOT = Path("/Users/tipanoii/doc/TA/code/seaweed/")
freq_tif = ROOT / "output_frequency_tif/2025_frequency_q1_q2_q4.tif"

sal_nc = Path("/Users/tipanoii/doc/TA/code/salinity/2025.nc")
temp_nc = Path("/Users/tipanoii/doc/TA/code/temp/2025.nc")
depth_tif = Path("/Users/tipanoii/doc/TA/code/depth/Depth_Batnas_Focal_35m.tif")

out_factor_tif = ROOT / "output_factor/factor_suitability_2025.tif"
out_factor_png = ROOT / "output_factor/factor_suitability_2025.png"

out_csv = ROOT / "distribusi_freq_d35.csv"

print("Analisis 2025\n")

# =========================================================
# 1. BACA RASTER FREKUENSI
# =========================================================
freq = rioxarray.open_rasterio(freq_tif).squeeze()

FREQ_NODATA = -999
freq = freq.where(freq != FREQ_NODATA)

print("Shape frequency:", freq.shape)
print("CRS frequency:", freq.rio.crs)

# =========================================================
# 2. BACA DATA SALINITAS & TEMP NETCDF
# =========================================================
ds = xr.open_dataset(sal_nc, engine="netcdf4")
print("Data variables salinitas:", list(ds.data_vars))
print("Coords salinitas:", list(ds.coords))
# ganti jika nama variabel bukan "sos"
sal = ds["so"]

print("===========================")

ds_temp = xr.open_dataset(temp_nc, engine="netcdf4")
print("Data variables temp:", list(ds_temp.data_vars))
print("Coords temp:", list(ds_temp.coords))
# ganti jika nama variabel bukan "sos"
temp = ds_temp["to"]
print()

# =========================================================
# 3. PILIH SALINITAS & TEMP PERMUKAAN
# CONTOH: rata-rata 1 tahun 2025
# =========================================================
cur_sal = sal.sel(time=slice("2025-01-01", "2026-01-01")).mean("time")
if "depth" in cur_sal.dims:
    cur_sal = cur_sal.sel(depth=0, method="nearest")
elif "deptht" in cur_sal.dims:
    cur_sal = cur_sal.sel(deptht=0, method="nearest")
    
cur_temp = temp.sel(time=slice("2025-01-01", "2026-01-01")).mean("time")
if "depth" in cur_temp.dims:
    cur_temp = cur_temp.sel(depth=0, method="nearest")
elif "deptht" in cur_temp.dims:
    cur_temp = cur_temp.sel(deptht=0, method="nearest")

# =========================================================
# 4. RAPIIKAN NAMA DIMENSI SPASIAL SALINITAS & TEMP
# =========================================================
rename_dict = {}

if "longitude" in cur_sal.dims:
    rename_dict["longitude"] = "x"
if "latitude" in cur_sal.dims:
    rename_dict["latitude"] = "y"
if "lon" in cur_sal.dims:
    rename_dict["lon"] = "x"
if "lat" in cur_sal.dims:
    rename_dict["lat"] = "y"

cur_sal = cur_sal.rename(rename_dict)
cur_sal = cur_sal.rio.set_spatial_dims(x_dim="x", y_dim="y")
cur_sal = cur_sal.rio.write_crs("EPSG:4326")

print("Shape salinity original:", cur_sal.shape)
print("CRS salinity original:", cur_sal.rio.crs)

print("===========================")

cur_temp = cur_temp.rename(rename_dict)
cur_temp = cur_temp.rio.set_spatial_dims(x_dim="x", y_dim="y")
cur_temp = cur_temp.rio.write_crs("EPSG:4326")

print("Shape temp original:", cur_temp.shape)
print("CRS temp original:", cur_temp.rio.crs)
print()

# =========================================================
# 5. REPROJECT SALINITAS & TEMP KE GRID FREKUENSI
# =========================================================
sal_on_freq = cur_sal.rio.reproject_match(
    freq,
    resampling=Resampling.nearest
)

temp_on_freq = cur_temp.rio.reproject_match(
    freq,
    resampling=Resampling.nearest
)

print("Shape salinity on frequency grid:", sal_on_freq.shape)
print("CRS salinity on frequency grid:", sal_on_freq.rio.crs)

print("===========================")

print("Shape temp on frequency grid:", temp_on_freq.shape)
print("CRS temp on frequency grid:", temp_on_freq.rio.crs)
print()

# =========================================================
# 6. BACA DATA DEPTH GEOTIFF
# =========================================================
print("====================================")
depth = rioxarray.open_rasterio(depth_tif).squeeze()

print("Shape depth original:", depth.shape)
print("CRS depth original:", depth.rio.crs)

# Jika depth punya nodata bawaan, buang
if depth.rio.nodata is not None:
    depth = depth.where(depth != depth.rio.nodata)

# Reproject depth ke grid frekuensi
depth_on_freq = depth.rio.reproject_match(
    freq,
    resampling=Resampling.nearest
)

print("Shape depth on frequency grid:", depth_on_freq.shape)
print("CRS depth on frequency grid:", depth_on_freq.rio.crs)
print()

# =========================================================
# 7. UBAH DEPTH MENJADI POSITIF (JIKA PERLU)
# Jika raster bathymetry Anda negatif, aktifkan abs()
# Jika sudah positif, baris ini tetap aman
# =========================================================
depth_on_freq = np.abs(depth_on_freq)

# =========================================================
# 8. AMBIL ARRAY
# =========================================================
freq_vals = freq.values
sal_vals = sal_on_freq.values
depth_vals = depth_on_freq.values
temp_vals = temp_on_freq.values

# Frekuensi yang ingin dihitung
# kalau ingin juga frekuensi 0, ubah jadi >= 0
unique_freqs = sorted([
    int(v) for v in np.unique(freq_vals[np.isfinite(freq_vals)])
    if v >= 1
])

print("Frekuensi yang ditemukan:", unique_freqs)

# =========================================================
# 9. BUAT BIN SALINITAS 0.5
# =========================================================
sal_valid_global = sal_vals[np.isfinite(sal_vals) & np.isfinite(freq_vals) & (freq_vals >= 1)]

if len(sal_valid_global) == 0:
    raise ValueError("Tidak ada nilai salinitas valid untuk frekuensi >= 1.")

'''
# batas general
sal_bin_width = 0.5
sal_min = np.floor(sal_valid_global.min() / sal_bin_width) * sal_bin_width
sal_max = np.ceil(sal_valid_global.max() / sal_bin_width) * sal_bin_width
'''

# batas rumput laut 
sal_bin_width = 0.5
sal_min = 28
sal_max = 34

sal_edges = np.arange(sal_min, sal_max + sal_bin_width, sal_bin_width)
sal_labels = [f"{sal_edges[i]:.1f} - {sal_edges[i+1]:.1f}" for i in range(len(sal_edges) - 1)]
print("Bin salinitas:", sal_labels)

# =========================================================
# 10. BUAT BIN DEPTH 0.5 M
# =========================================================
depth_valid_global = depth_vals[np.isfinite(depth_vals) & np.isfinite(freq_vals) & (freq_vals >= 1)]

if len(depth_valid_global) == 0:
    raise ValueError("Tidak ada nilai depth valid untuk frekuensi >= 1.")

'''
# batas general
depth_bin_width = 0.5
depth_min = np.floor(depth_valid_global.min() / depth_bin_width) * depth_bin_width
depth_max = np.ceil(depth_valid_global.max() / depth_bin_width) * depth_bin_width
'''

# batas rumput laut 
depth_bin_width = 0.5
depth_min = 0.5
depth_max = 5

depth_edges = np.arange(depth_min, depth_max + depth_bin_width, depth_bin_width)
depth_labels = [f"{depth_edges[i]:.1f} - {depth_edges[i+1]:.1f}" for i in range(len(depth_edges) - 1)]
print("Bin depth:", depth_labels)

# =========================================================
# 10. BUAT BIN TEMP 0.5 C
# =========================================================
temp_valid_global = temp_vals[np.isfinite(temp_vals) & np.isfinite(freq_vals) & (freq_vals >= 1)]

if len(temp_valid_global) == 0:
    raise ValueError("Tidak ada nilai depth valid untuk frekuensi >= 1.")

'''
# batas general
temp_bin_width = 0.5
temp_min = np.floor(depth_valid_global.min() / depth_bin_width) * depth_bin_width
temp_max = np.ceil(depth_valid_global.max() / depth_bin_width) * depth_bin_width
'''

# batas rumput laut 
temp_bin_width = 0.5
temp_min = 26
temp_max = 30

temp_edges = np.arange(temp_min, temp_max + temp_bin_width, temp_bin_width)
temp_labels = [f"{temp_edges[i]:.1f} - {temp_edges[i+1]:.1f}" for i in range(len(temp_edges) - 1)]
print("Bin temp:", temp_labels)
print()

# =========================================================
# 11. HITUNG DISTRIBUSI GABUNGAN
# output: frequency, salinity_interval, depth_interval, count_pixels
# =========================================================
all_results = []

for f in unique_freqs:
    mask_f = (freq_vals == f)

    total_pixels = int(np.count_nonzero(mask_f))
    print(f"\nMemproses frekuensi {f} | total piksel: {total_pixels}")

    sal_this = sal_vals[mask_f]
    depth_this = depth_vals[mask_f]
    temp_this = temp_vals[mask_f]

    # hitung per piksel
    rows = []
    for s, d, t in zip(sal_this, depth_this, temp_this):
        # ---------- salinity label ----------
        if np.isfinite(s):
            s_adj = s
            if s_adj == sal_max:
                s_adj = np.nextafter(sal_max, -np.inf)

            s_bin = pd.cut(
                [s_adj],
                bins=sal_edges,
                labels=sal_labels,
                right=False,
                include_lowest=True
            )[0]

            if pd.isna(s_bin):
                sal_label = "out of range"
            else:
                sal_label = str(s_bin)
        else:
            sal_label = "no data avail"

        # ---------- depth label ----------
        if np.isfinite(d):
            d_adj = d
            if d_adj == depth_max:
                d_adj = np.nextafter(depth_max, -np.inf)

            d_bin = pd.cut(
                [d_adj],
                bins=depth_edges,
                labels=depth_labels,
                right=False,
                include_lowest=True
            )[0]

            if pd.isna(d_bin):
                depth_label = "out of range"
            else:
                depth_label = str(d_bin)
        else:
            depth_label = "no data avail"
        
        # ---------- depth temp ----------
        if np.isfinite(t):
            t_adj = t
            if t_adj == temp_max:
                t_adj = np.nextafter(temp_max, -np.inf)

            t_bin = pd.cut(
                [t_adj],
                bins=temp_edges,
                labels=temp_labels,
                right=False,
                include_lowest=True
            )[0]

            if pd.isna(t_bin):
                temp_label = "out of range"
            else:
                temp_label = str(t_bin)
        else:
            temp_label = "no data avail"

        rows.append((f, depth_label, sal_label, temp_label))

    df_f = pd.DataFrame(rows, columns=["frequency", "depth","salinity", "temp"])

    count_f = (
        df_f.groupby(["frequency", "depth", "salinity", "temp"])
        .size()
        .reset_index(name="count_pixels")
    )

    all_results.append(count_f)

# =========================================================
# 12. GABUNGKAN SEMUA HASIL
# =========================================================
final_table = pd.concat(all_results, ignore_index=True)

# urutkan
final_table = final_table.sort_values(
    by=["frequency", "depth", "salinity", "temp"]
).reset_index(drop=True)

print("\nHasil distribusi gabungan:")
print(final_table)

# =========================================================
# 13. SIMPAN CSV
# =========================================================
final_table.to_csv(out_csv, index=False)
print(f"\nCSV tersimpan: {out_csv}")

# =========================================================
# MASK POTENSIAL DASAR
# contoh: hanya piksel hasil pemetaan yang potensial
# jika mau lebih ketat, bisa ganti freq_vals >= 2
# =========================================================
potential_mask = np.isfinite(freq_vals) & (freq_vals >= 1)

# =========================================================
# CEK KETERSEDIAAN DATA LINGKUNGAN
# =========================================================
sal_available = np.isfinite(sal_vals)
depth_available = np.isfinite(depth_vals)
temp_available = np.isfinite(temp_vals)

all_env_available = sal_available & depth_available & temp_available
any_env_missing = ~(all_env_available)

# =========================================================
# CEK KESESUAIAN MASING-MASING FAKTOR
# =========================================================
sal_ok = sal_available & (sal_vals >= sal_min) & (sal_vals <= sal_max)
depth_ok = depth_available & (depth_vals >= depth_min) & (depth_vals <= depth_max)
temp_ok = temp_available & (temp_vals >= temp_min) & (temp_vals <= temp_max)

# =========================================================
# HITUNG JUMLAH FAKTOR YANG TERPENUHI
# =========================================================
factor_count = (
    sal_ok.astype(np.int16)
    + depth_ok.astype(np.int16)
    + temp_ok.astype(np.int16)
)

# =========================================================
# KELAS FINAL UNTUK VISUALISASI
# -999 : di luar area potensial / background
# -998 : data lingkungan tidak lengkap
# 0    : tidak memenuhi faktor
# 1    : memenuhi 1 faktor
# 2    : memenuhi 2 faktor
# 3    : memenuhi 3 faktor
# =========================================================
VIS_NODATA = -999
VIS_ENV_NODATA = -998

factor_class = np.full(freq_vals.shape, VIS_NODATA, dtype=np.int16)

# area potensial tapi data lingkungan tidak lengkap
factor_class[potential_mask & any_env_missing] = VIS_ENV_NODATA

# area potensial dan data lingkungan lengkap
valid_eval_mask = potential_mask & all_env_available
factor_class[valid_eval_mask] = factor_count[valid_eval_mask]

summary_df = pd.DataFrame({
    "frequency": freq_vals.ravel(),
    "factor_class": factor_class.ravel()
})

summary_df = summary_df[
    np.isfinite(summary_df["frequency"]) &
    (summary_df["frequency"] >= 1) &
    (summary_df["factor_class"] >= 0)
]

summary_count = (
    summary_df.groupby(["frequency", "factor_class"])
    .size()
    .reset_index(name="count_pixels")
)

print("\nRingkasan per frequency dan jumlah faktor:")
print(summary_count)

# =========================================================
# SIMPAN RASTER
# =========================================================
with rasterio.open(freq_tif) as src:
    profile = src.profile.copy()

profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=VIS_NODATA,
    compress="lzw"
)

with rasterio.open(out_factor_tif, "w", **profile) as dst:
    dst.write(factor_class, 1)

print(f"GeoTIFF faktor tersimpan: {out_factor_tif}")

# =========================================================
# SIMPAN PNG
# =========================================================
def save_factor_png(class_arr, out_png, title="Kesesuaian Faktor Habitat", show_plot=False):
    h, w = class_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # default hitam = di luar area potensial
    rgb[:, :] = [0, 0, 0]

    # data lingkungan tidak lengkap = abu-abu
    rgb[class_arr == -998] = [160, 160, 160]

    # 0 faktor = putih
    rgb[class_arr == 0] = [255, 255, 255]

    # 1 faktor = kuning
    rgb[class_arr == 1] = [255, 204, 102]

    # 2 faktor = oranye
    rgb[class_arr == 2] = [255, 102, 0]

    # 3 faktor = merah
    rgb[class_arr == 3] = [204, 0, 0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array([255, 255, 255]) / 255.0, edgecolor="black", label="0 faktor"),
        Patch(facecolor=np.array([255, 204, 102]) / 255.0, edgecolor="black", label="1 faktor"),
        Patch(facecolor=np.array([255, 102, 0]) / 255.0, edgecolor="black", label="2 faktor"),
        Patch(facecolor=np.array([204, 0, 0]) / 255.0, edgecolor="black", label="3 faktor"),
        Patch(facecolor=np.array([160, 160, 160]) / 255.0, edgecolor="black", label="data lingkungan tidak lengkap"),
        Patch(facecolor=np.array([0, 0, 0]) / 255.0, edgecolor="black", label="di luar area potensial"),
    ]

    ax.legend(
        handles=legend_elements,
        title="Kelas",
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

save_factor_png(
    class_arr=factor_class,
    out_png=out_factor_png,
    title="Area Potensial Berdasarkan 3, 2, dan 1 Faktor",
    show_plot=True
)