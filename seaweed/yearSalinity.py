#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:01:43 2026

@author: tipanoii
"""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

file = Path("/Users/tipanoii/doc/TA/code/seaweed/salinity_2024.nc")
ds = xr.open_dataset(str(file), engine="netcdf4")

print("dims:", ds.dims)
print("coords:", list(ds.coords))
print("data_vars:", list(ds.data_vars))

candidate_vars = ["sos", "so", "sss", "salinity"]
sal_var = None
for v in candidate_vars:
    if v in ds.data_vars:
        sal_var = v
        break
if sal_var is None:
    raise ValueError(
        f"Tidak menemukan variabel salinitas. Data variables yang ada: {list(ds.data_vars)}"
    )
print("Variabel salinitas yang dipakai:", sal_var)

sal = ds[sal_var]
print(sal)

# =========================================
# Ambil 1 layer 2D untuk visualisasi
#    - jika ada time, ambil time pertama
#    - jika ada depth, ambil depth terdekat ke 0 m
# =========================================
sal_plot = sal

if "time" in sal_plot.dims:
    sal_plot = sal_plot.isel(time=0)

if "depth" in sal_plot.dims:
    sal_plot = sal_plot.sel(depth=0, method="nearest")
elif "deptht" in sal_plot.dims:
    sal_plot = sal_plot.sel(deptht=0, method="nearest")
    
# =========================================
# Visualisasi salinitas
# =========================================
plt.figure(figsize=(8, 6))
sal_plot.plot.contourf(
    cmap="viridis",
    levels=20,
    cbar_kwargs={"label": "Salinitas"}
)
plt.title("Visualisasi Salinitas")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()