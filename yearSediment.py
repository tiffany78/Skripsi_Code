#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:35:21 2026

@author: tipanoii
"""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

# Folder berisi file .nc
folder = Path("/Users/tipanoii/doc/TA/code/sediment")

# Ambil semua file .nc
nc_files = list(folder.glob("*.nc"))

# Loop setiap file
for file in nc_files:
    print(f"\nProcessing: {file.name}")
    
    ds = xr.open_dataset(str(file), engine="netcdf4")
    
    candidate_vars = ["SPM"]
    sal_var = None
    for v in candidate_vars:
        if v in ds.data_vars:
            sal_var = v
            break

    if sal_var is None:
        print(f"⚠️ Skip {file.name} (tidak ada variabel temp)")
        continue

    print("Variabel temp:", sal_var)
    
    sal = ds[sal_var]

    # Ambil layer 2D
    sal_plot = sal

    if "time" in sal_plot.dims:
        sal_plot = sal_plot.isel(time=0)

    if "depth" in sal_plot.dims:
        sal_plot = sal_plot.sel(depth=0, method="nearest")
    elif "deptht" in sal_plot.dims:
        sal_plot = sal_plot.sel(deptht=0, method="nearest")

    # Plot
    plt.figure(figsize=(8, 6))
    sal_plot.plot.contourf(
        cmap="viridis",
        levels=20,
        cbar_kwargs={"label": "Sediment"}
    )
    
    plt.title(f"Sediment - {file.stem}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # Save PNG (nama sama dengan file .nc)
    output_path = folder / f"{file.stem}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {output_path}")