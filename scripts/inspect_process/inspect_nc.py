import xarray as xr

ds = xr.open_dataset("../data/process/era5/1993_1.nc")

print(ds)
print(ds.dims)
print(ds.coords)