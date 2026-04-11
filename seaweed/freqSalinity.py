#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:32:00 2026

@author: tipanoii
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

# =========================================================
# PATH
# =========================================================
freq_tif = Path("/Users/tipanoii/doc/TA/code/seaweed/output_frequency_tif/2024_frequency_q1_q4.tif")
sal_nc = Path("/Users/tipanoii/doc/TA/code/seaweed/salinity_2024.nc")

# =========================================================
# 1. BACA RASTER FREKUENSI
# =========================================================
freq = rioxarray.open_rasterio(freq_tif).squeeze()

# ganti jika nodata Anda berbeda
FREQ_NODATA = -999
freq = freq.where(freq != FREQ_NODATA)

print("Shape frequency:", freq.shape)
print("CRS frequency:", freq.rio.crs)

# mask piksel frekuensi = 3
mask_freq3 = (freq == 3)

jumlah_freq3 = int(mask_freq3.sum().item())
print("Jumlah piksel frekuensi 3:", jumlah_freq3)

# =========================================================
# 2. BACA DATA SALINITAS NETCDF
# =========================================================
ds = xr.open_dataset(sal_nc, engine="netcdf4")

print("Data variables:", list(ds.data_vars))
print("Coords:", list(ds.coords))

# ganti nama variabel jika bukan 'sos'
sal = ds["sos"]

# =========================================================
# 3. PILIH SALINITAS PERMUKAAN TAHUN 2024
# =========================================================
sal_2024 = sal.sel(time=slice("2024-01-01", "2024-12-31")).mean("time")

# kalau ada dimensi depth / deptht, ambil permukaan
if "depth" in sal_2024.dims:
    sal_2024 = sal_2024.sel(depth=0, method="nearest")
elif "deptht" in sal_2024.dims:
    sal_2024 = sal_2024.sel(deptht=0, method="nearest")

# =========================================================
# 4. RAPIIKAN NAMA DIMENSI SPASIAL
# =========================================================
rename_dict = {}

if "longitude" in sal_2024.dims:
    rename_dict["longitude"] = "x"
if "latitude" in sal_2024.dims:
    rename_dict["latitude"] = "y"
if "lon" in sal_2024.dims:
    rename_dict["lon"] = "x"
if "lat" in sal_2024.dims:
    rename_dict["lat"] = "y"

sal_2024 = sal_2024.rename(rename_dict)

sal_2024 = sal_2024.rio.set_spatial_dims(x_dim="x", y_dim="y")
sal_2024 = sal_2024.rio.write_crs("EPSG:4326")

print("Shape salinity original:", sal_2024.shape)
print("CRS salinity original:", sal_2024.rio.crs)

# =========================================================
# 5. REPROJECT SALINITAS KE GRID FREKUENSI
# =========================================================
# nearest = mempertahankan nilai asli grid salinitas
# kalau mau lebih halus bisa ganti ke Resampling.bilinear
sal_on_freq = sal_2024.rio.reproject_match(
    freq,
    resampling=Resampling.nearest
)

print("Shape salinity on frequency grid:", sal_on_freq.shape)
print("CRS salinity on frequency grid:", sal_on_freq.rio.crs)

# =========================================================
# 6. AMBIL NILAI SALINITAS HANYA PADA PIKSEL FREKUENSI 3
# =========================================================
freq_vals = freq.values
sal_vals = sal_on_freq.values

valid_mask = (
    (freq_vals == 3) &
    np.isfinite(sal_vals)
)

sal_freq3 = sal_vals[valid_mask]

print("Jumlah piksel freq=3 yang punya nilai salinitas:", len(sal_freq3))

# =========================================================
# 7. FILTER RENTANG SALINITAS 33 - 35
# =========================================================
sal_min = 33.0
sal_max = 35.0

sal_freq3_range = sal_freq3[(sal_freq3 >= sal_min) & (sal_freq3 <= sal_max)]

print("Jumlah piksel freq=3 pada rentang 33-35:", len(sal_freq3_range))

# =========================================================
# 8. HITUNG COUNT PER INTERVAL 0.1
# Label 33.0 berarti interval [33.0, 33.1)
# Label 33.1 berarti interval [33.1, 33.2), dst
# =========================================================
edges = np.round(np.arange(33.0, 35.0 + 0.1, 0.1), 1)
labels = [f"{edges[i]:.1f}" for i in range(len(edges) - 1)]

# supaya nilai tepat 35.0 tetap masuk bin terakhir
sal_freq3_range = np.where(
    sal_freq3_range == 35.0,
    np.nextafter(35.0, -np.inf),
    sal_freq3_range
)

bins = pd.cut(
    sal_freq3_range,
    bins=edges,
    labels=labels,
    right=False,
    include_lowest=True
)

count_table = (
    pd.Series(bins)
    .value_counts(sort=False)
    .reindex(labels, fill_value=0)
    .reset_index()
)

count_table.columns = ["salinity_interval", "count_pixels"]

print("\nDistribusi piksel frekuensi 3 berdasarkan salinitas:")
print(count_table)

# =========================================================
# 9. OPTIONAL: SIMPAN CSV
# =========================================================
out_csv = Path("/Users/tipanoii/doc/TA/code/seaweed/distribusi_freq3_salinitas_2024.csv")
count_table.to_csv(out_csv, index=False)
print(f"\nCSV tersimpan: {out_csv}")