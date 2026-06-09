#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:38:16 2026

@author: tipanoii
"""

from pathlib import Path
import re
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
ROOT_FREQ = ""
MODE = "reef" # "seaweed" or "reef"

if MODE == "seaweed":
    print("\n============== ANALISIS RUMPUT LAUT ========================")
    ROOT = Path("D:/TA/code/seaweed")
    ROOT_FREQ = ROOT / "output"
    
    # batas habitat rumput laut
    sal_bin_width = 0.5
    sal_min = 28
    sal_max = 34

    depth_bin_width = 0.5
    depth_min = 3
    depth_max = 5

    temp_bin_width = 0.5
    temp_min = 24
    temp_max = 30
    
    sen_bin_width = 5
    sen_min = 0
    sen_max = 25
elif MODE == "reef":
    print("\n============== ANALISIS TERUMBU KARANG ========================")
    ROOT = Path("D:/TA/code/reef/")
    ROOT_FREQ = ROOT / "output"
    
    # batas habitat terumbu karang
    sal_bin_width = 0.5
    sal_min = 30
    sal_max = 35

    depth_bin_width = 0.5
    depth_min = 4
    depth_max = 8

    temp_bin_width = 0.5
    temp_min = 23
    temp_max = 30
    
    sen_bin_width = 5
    sen_min = 0
    sen_max = 20
else:
    raise SystemExit("Tipe Analisis Tidak Sesuai")

ROOT_SAL  = Path("D:/TA/code/salinity/")
ROOT_TEMP = Path("D:/TA/code/temp/")
ROOT_DEPTH = Path("D:/TA/code/depth/")
ROOT_SEN = Path("D:/TA/code/sediment/")

OUT_FACTOR_DIR = ROOT / "output_potensial"
OUT_CSV_DIR = OUT_FACTOR_DIR

OUT_FACTOR_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# INPUT 1 DATA FREKUENSI DAN 2 DATA DEPTH
# =========================================================
YEAR_ANALYSIS = "2025"

# Sesuaikan nama file frekuensi 2025 di folder ROOT_FREQ
FREQ_TIF = ROOT_FREQ / "2025_frequency_q1_q2_q3_q4.tif"

# Mode: "highest" = proses hanya frekuensi tertinggi, misalnya hanya 4
FREQUENCY_MODE = "highest"

# Depth yang diproses hanya 2 file ini
DEPTH_TIFS = [
    ROOT_DEPTH / "Depth_35m_2025.tif",
    ROOT_DEPTH / "Depth_10m_2025.tif",
]

# =========================================================
# PARAMETER GLOBAL
# =========================================================
FREQ_NODATA = -999
VIS_NODATA = -999
VIS_ENV_NODATA = -998

# =========================================================
# HELPER
# =========================================================
def extract_year(path_obj):
    """
    Ambil tahun 4 digit dari nama file.
    contoh:
    2025_frequency_q1_q2_q4.tif -> 2025
    """
    m = re.search(r"(20\d{2})", path_obj.stem)
    if not m:
        raise ValueError(f"Tahun tidak ditemukan pada nama file: {path_obj.name}")
    return m.group(1)

def clean_name(path_obj):
    """
    Nama aman untuk output file.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "_", path_obj.stem).strip("_")

def standardize_xy(da):
    rename_dict = {}

    if "longitude" in da.dims:
        rename_dict["longitude"] = "x"
    if "latitude" in da.dims:
        rename_dict["latitude"] = "y"

    da = da.rename(rename_dict)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326")
    return da

def save_factor_png(class_arr, out_png, title="Kesesuaian Faktor Habitat", show_plot=False):
    h, w = class_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # default hitam = di luar area potensial
    rgb[:, :] = [0, 0, 0]

    # data lingkungan tidak lengkap = abu-abu
    rgb[class_arr == -998] = [160, 160, 160]

    # 0 faktor = putih
    rgb[class_arr == 0] = [255, 255, 255]
    
    # 1 faktor = putih
    rgb[class_arr == 1] = [255, 255, 200]

    # 2 faktor = kuning
    rgb[class_arr == 2] = [255, 204, 102]

    # 3 faktor = oranye
    rgb[class_arr == 3] = [255, 102, 0]

    # 4 faktor = merah
    rgb[class_arr == 4] = [204, 0, 0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array([255, 255, 255]) / 255.0, edgecolor="black", label="0 faktor"),
        Patch(facecolor=np.array([255, 255, 200]) / 255.0, edgecolor="black", label="1 faktor"),
        Patch(facecolor=np.array([255, 204, 102]) / 255.0, edgecolor="black", label="2 faktor"),
        Patch(facecolor=np.array([255, 102, 0]) / 255.0, edgecolor="black", label="3 faktor"),
        Patch(facecolor=np.array([204, 0, 0]) / 255.0, edgecolor="black", label="4 faktor"),
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

def load_external_for_year(year, freq):
    """
    Load salinitas dan suhu untuk 1 tahun, lalu reproject ke grid freq.
    """
    sal_nc = ROOT_SAL / f"{year}.nc"
    temp_nc = ROOT_TEMP / f"{year}.nc"
    sen_nc = ROOT_SEN / f"{year}.nc"

    if not sal_nc.exists():
        raise FileNotFoundError(f"File salinitas tidak ditemukan: {sal_nc}")
    if not temp_nc.exists():
        raise FileNotFoundError(f"File suhu tidak ditemukan: {temp_nc}")
    if not sen_nc.exists():
        raise FileNotFoundError(f"File suhu tidak ditemukan: {sen_nc}")

    ds_sal = xr.open_dataset(sal_nc, engine="netcdf4")
    ds_temp = xr.open_dataset(temp_nc, engine="netcdf4")
    ds_sen = xr.open_dataset(sen_nc, engine="netcdf4")

    # ganti jika variabel berbeda
    sal = ds_sal["sos"]
    temp = ds_temp["to"]
    sen = ds_sen["SPM"]

    # rata-rata tahunan
    cur_sal = sal.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
    cur_sal = cur_sal.sel(depth=0, method="nearest")
    
    cur_temp = temp.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
    cur_temp = cur_temp.sel(depth=0, method="nearest")
    
    cur_sen = sen.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")

    cur_sal = standardize_xy(cur_sal)
    cur_temp = standardize_xy(cur_temp)
    cur_sen = standardize_xy(cur_sen)
    
    ds_sal.close()
    ds_temp.close()
    ds_sen.close()

    sal_on_freq = cur_sal.rio.reproject_match(freq, resampling=Resampling.nearest)
    temp_on_freq = cur_temp.rio.reproject_match(freq, resampling=Resampling.nearest)
    sen_on_freq = cur_sen.rio.reproject_match(freq, resampling=Resampling.nearest)

    return sal_on_freq, temp_on_freq, sen_on_freq

def get_selected_frequencies(freq_vals):
    """
    FREQUENCY_MODE = "highest"
        -> hanya frekuensi tertinggi yang dipakai.
    """
    available_freqs = sorted([
        int(v) for v in np.unique(freq_vals[np.isfinite(freq_vals)])
        if v >= 1
    ])

    if not available_freqs:
        raise ValueError("Tidak ada nilai frekuensi valid >= 1 pada raster.")

    return [max(available_freqs)]

def process_one_combination(freq_tif, depth_tif):
    year = extract_year(freq_tif)
    depth_name = clean_name(depth_tif)

    # nama output dibuat memuat mode frekuensi
    out_factor_tif = OUT_FACTOR_DIR / f"factor_suitability_{year}_{depth_name}_{FREQUENCY_MODE}.tif"
    out_factor_png = OUT_FACTOR_DIR / f"factor_suitability_{year}_{depth_name}_{FREQUENCY_MODE}.png"
    out_csv = OUT_CSV_DIR / f"distribusi_freq_{year}_{depth_name}_{FREQUENCY_MODE}.csv"

    print("\n===================================================")
    print(f"Analisis tahun {year}")
    print(f"Freq  : {freq_tif.name}")
    print(f"Depth : {depth_tif.name}")
    print("===================================================")

    # =========================================================
    # 1. BACA RASTER FREKUENSI
    # =========================================================
    freq = rioxarray.open_rasterio(freq_tif).squeeze()
    freq = freq.where(freq != FREQ_NODATA)

    print("Shape frequency:", freq.shape)
    print("CRS frequency:", freq.rio.crs)
    print()

    # =========================================================
    # 2. LOAD SALINITAS & TEMP SESUAI TAHUN
    # =========================================================
    sal_on_freq, temp_on_freq, sen_on_freq = load_external_for_year(year, freq)

    print("Shape salinity on frequency grid:", sal_on_freq.shape)
    print("CRS salinity on frequency grid:", sal_on_freq.rio.crs)
    print()
    print("Shape temp on frequency grid:", temp_on_freq.shape)
    print("CRS temp on frequency grid:", temp_on_freq.rio.crs)
    print()
    print("Shape sediment on frequency grid:", sen_on_freq.shape)
    print("CRS sediment on frequency grid:", sen_on_freq.rio.crs)
    print()

    # =========================================================
    # 3. BACA DEPTH
    # =========================================================
    depth = rioxarray.open_rasterio(depth_tif).squeeze()

    print("Shape depth original:", depth.shape)
    print("CRS depth original:", depth.rio.crs)

    if depth.rio.nodata is not None:
        depth = depth.where(depth != depth.rio.nodata)

    depth_on_freq = depth.rio.reproject_match(freq, resampling=Resampling.nearest)
    depth_on_freq = np.abs(depth_on_freq)

    print("Shape depth on frequency grid:", depth_on_freq.shape)
    print("CRS depth on frequency grid:", depth_on_freq.rio.crs)
    print()

    # =========================================================
    # 4. AMBIL ARRAY
    # =========================================================
    freq_vals = freq.values
    sal_vals = sal_on_freq.values
    depth_vals = depth_on_freq.values
    temp_vals = temp_on_freq.values
    sen_vals = sen_on_freq.values

    available_freqs = sorted([
        int(v) for v in np.unique(freq_vals[np.isfinite(freq_vals)])
        if v >= 1
    ])
    
    selected_freqs = get_selected_frequencies(freq_vals)
    
    print("Frekuensi yang ditemukan:", available_freqs)
    print("Mode frekuensi:", FREQUENCY_MODE)
    print("Frekuensi yang dianalisis:", selected_freqs)

    # =========================================================
    # 5. BUAT BIN
    # =========================================================
    sal_edges = np.arange(sal_min, sal_max + sal_bin_width, sal_bin_width)
    sal_labels = [f"{sal_edges[i]:.1f} - {sal_edges[i+1]:.1f}" for i in range(len(sal_edges) - 1)]

    depth_edges = np.arange(depth_min, depth_max + depth_bin_width, depth_bin_width)
    depth_labels = [f"{depth_edges[i]:.1f} - {depth_edges[i+1]:.1f}" for i in range(len(depth_edges) - 1)]

    temp_edges = np.arange(temp_min, temp_max + temp_bin_width, temp_bin_width)
    temp_labels = [f"{temp_edges[i]:.1f} - {temp_edges[i+1]:.1f}" for i in range(len(temp_edges) - 1)]
    
    sen_edges = np.arange(sen_min, sen_max + sen_bin_width, sen_bin_width)
    sen_labels = [f"{sen_edges[i]:.1f} - {sen_edges[i+1]:.1f}" for i in range(len(sen_edges) - 1)]

    # =========================================================
    # 6. HITUNG DISTRIBUSI GABUNGAN
    # =========================================================
    all_results = []

    for f in selected_freqs:
        mask_f = (freq_vals == f)
        total_pixels = int(np.count_nonzero(mask_f))
        print(f"Memproses frekuensi {f} | total piksel: {total_pixels}")

        sal_this = sal_vals[mask_f]
        depth_this = depth_vals[mask_f]
        temp_this = temp_vals[mask_f]
        sen_this = sen_vals[mask_f]

        rows = []
        for s, d, t, se in zip(sal_this, depth_this, temp_this, sen_this):
            # salinity
            if np.isfinite(s):
                s_adj = np.nextafter(sal_max, -np.inf) if s == sal_max else s
                s_bin = pd.cut(
                    [s_adj], bins=sal_edges, labels=sal_labels,
                    right=False, include_lowest=True
                )[0]
                sal_label = "out of range" if pd.isna(s_bin) else str(s_bin)
            else:
                sal_label = "no data avail"

            # depth
            if np.isfinite(d):
                d_adj = np.nextafter(depth_max, -np.inf) if d == depth_max else d
                d_bin = pd.cut(
                    [d_adj], bins=depth_edges, labels=depth_labels,
                    right=False, include_lowest=True
                )[0]
                depth_label = "out of range" if pd.isna(d_bin) else str(d_bin)
            else:
                depth_label = "no data avail"

            # temp
            if np.isfinite(t):
                t_adj = np.nextafter(temp_max, -np.inf) if t == temp_max else t
                t_bin = pd.cut(
                    [t_adj], bins=temp_edges, labels=temp_labels,
                    right=False, include_lowest=True
                )[0]
                temp_label = "out of range" if pd.isna(t_bin) else str(t_bin)
            else:
                temp_label = "no data avail"
                
            # sediment
            if np.isfinite(se):
                se_adj = np.nextafter(sen_max, -np.inf) if se == sen_max else se
                se_bin = pd.cut(
                    [se_adj], bins=sen_edges, labels=sen_labels,
                    right=False, include_lowest=True
                )[0]
                sen_label = "out of range" if pd.isna(se_bin) else str(se_bin)
            else:
                sen_label = "no data avail"

            rows.append((f, depth_label, sal_label, temp_label, sen_label))

        df_f = pd.DataFrame(rows, columns=["frequency", "depth", "salinity", "temp", "sediment"])

        count_f = (
            df_f.groupby(["frequency", "depth", "salinity", "temp", "sediment"])
            .size()
            .reset_index(name="count_pixels")
        )

        all_results.append(count_f)

    final_table = pd.concat(all_results, ignore_index=True)
    final_table = final_table.sort_values(
        by=["frequency", "depth", "salinity", "temp", "sediment"]
    ).reset_index(drop=True)

    final_table["year"] = year
    final_table["freq_file"] = freq_tif.name
    final_table["depth_file"] = depth_tif.name

    final_table.to_csv(out_csv, index=False)
    print(f"CSV tersimpan: {out_csv}")

    # =========================================================
    # 7. HITUNG FACTOR CLASS
    # =========================================================
    potential_mask = np.isfinite(freq_vals) & np.isin(freq_vals, selected_freqs)

    sal_available = np.isfinite(sal_vals)
    depth_available = np.isfinite(depth_vals)
    temp_available = np.isfinite(temp_vals)
    sen_available = np.isfinite(sen_vals)

    all_env_available = sal_available & depth_available & temp_available & sen_available
    any_env_missing = ~all_env_available

    sal_ok = sal_available & (sal_vals >= sal_min) & (sal_vals <= sal_max)
    depth_ok = depth_available & (depth_vals >= depth_min) & (depth_vals <= depth_max)
    temp_ok = temp_available & (temp_vals >= temp_min) & (temp_vals <= temp_max)
    sen_ok = sen_available & (sen_vals >= sen_min) & (sen_vals <= sen_max)

    factor_count = (
        sal_ok.astype(np.int16)
        + depth_ok.astype(np.int16)
        + temp_ok.astype(np.int16)
        + sen_ok.astype(np.int16)
    )

    factor_class = np.full(freq_vals.shape, VIS_NODATA, dtype=np.int16)
    factor_class[potential_mask & any_env_missing] = VIS_ENV_NODATA

    valid_eval_mask = potential_mask & all_env_available
    factor_class[valid_eval_mask] = factor_count[valid_eval_mask]

    # =========================================================
    # 8. SIMPAN RASTER
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
        dst.write(factor_class.astype(np.int16), 1)

    print(f"GeoTIFF faktor tersimpan: {out_factor_tif}")

    # =========================================================
    # 9. SIMPAN PNG
    # =========================================================
    save_factor_png(
        class_arr=factor_class,
        out_png=out_factor_png,
        title=f"Area Potensial {year} - {depth_name} - {selected_freqs}",
        show_plot=False
    )

# =========================================================
# MAIN LOOP UNTUK 1 FREKUENSI 2025 DAN 2 DEPTH
# =========================================================
if not FREQ_TIF.exists():
    raise FileNotFoundError(f"File frekuensi tidak ditemukan: {FREQ_TIF}")

for depth_tif in DEPTH_TIFS:
    if not depth_tif.exists():
        raise FileNotFoundError(f"File depth tidak ditemukan: {depth_tif}")

print("File frequency yang diproses:")
print(" -", FREQ_TIF.name)

print("\nFile depth yang diproses:")
for d in DEPTH_TIFS:
    print(" -", d.name)

for depth_tif in DEPTH_TIFS:
    try:
        process_one_combination(FREQ_TIF, depth_tif)
    except Exception as e:
        print(f"Gagal memproses kombinasi {FREQ_TIF.name} x {depth_tif.name}")
        print("Error:", e)