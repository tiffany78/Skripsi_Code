# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:35:52 2026

@author: Tiffany
"""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_SAL  = Path("D:/TA/code/salinity")
ROOT_TEMP = Path("D:/TA/code/temp")
ROOT_SEN = Path("D:/TA/code/sediment")
year = 2025

sal_nc = ROOT_SAL / f"{year}.nc"
temp_nc = ROOT_TEMP / f"{year}.nc"
sen_nc = ROOT_SEN / f"{year}.nc"

def makePlot(data, label, path):
    # Plot
    plt.figure(figsize=(8, 6))
    data.plot.contourf(
        cmap="viridis",
        levels=20,
        cbar_kwargs={"label": f"{label}"}
    )
    
    plt.title(f"{label} - 2025")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # Save PNG (nama sama dengan path dari .nc)
    output_path = path / f"{label}.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved: {output_path}")
    
    
'''
Pengecekan ketersediaan file 
'''
if not sal_nc.exists():
    raise FileNotFoundError(f"File salinitas tidak ditemukan: {sal_nc}")
if not temp_nc.exists():
    raise FileNotFoundError(f"File suhu tidak ditemukan: {temp_nc}")
if not sen_nc.exists():
    raise FileNotFoundError(f"File suhu tidak ditemukan: {sen_nc}")

ds_sal = xr.open_dataset(sal_nc, engine="netcdf4")
ds_temp = xr.open_dataset(temp_nc, engine="netcdf4")
ds_sen = xr.open_dataset(sen_nc, engine="netcdf4")

# Jenis Variabel yang digunakan
sal = ds_sal["sos"]
temp = ds_temp["to"]
sen = ds_sen["SPM"]

# rata-rata tahunan
cur_sal = sal.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
print(cur_sal)
cur_sal = cur_sal.sel(depth=0)
makePlot(cur_sal, "Salinitas", ROOT_SAL)

cur_temp = temp.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
print(cur_sal)
cur_temp = cur_temp.sel(depth=0)
makePlot(cur_temp, "Temperature", ROOT_TEMP)

cur_sen = sen.sel(time=slice(f"{year}-01-01", f"{int(year)+1}-01-01")).mean("time")
print(cur_sen)
makePlot(cur_sen, "Sediment", ROOT_SEN)