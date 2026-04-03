import argparse
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import xesmf as xe
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: xesmf. Install with `pip install xesmf`."
    ) from exc


def _normalize_lon(lon):
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _ensure_ascending(ds, coord):
    if coord in ds.coords:
        values = ds[coord].values
        if values[0] > values[-1]:
            ds = ds.sortby(coord)
    return ds


def _build_target_grid(glo12_path):
    ds = xr.open_dataset(glo12_path, decode_times=False)
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()
    grid = xr.Dataset(
        {
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        }
    )
    return grid


def _select_spatial_vars(ds):
    spatial_vars = []
    for name, da in ds.data_vars.items():
        if "lon" in da.dims and "lat" in da.dims:
            spatial_vars.append(name)
    if not spatial_vars:
        raise ValueError("No variables found with lon/lat dims")
    return ds[spatial_vars]


def regrid_era5_to_glo12(era5_path, glo12_path, output_path, var_name=None, method="bilinear"):
    src = xr.open_dataset(era5_path, decode_times=False)

    lon_name = "longitude" if "longitude" in src.coords else "lon"
    lat_name = "latitude" if "latitude" in src.coords else "lat"

    src = src.rename({lon_name: "lon", lat_name: "lat"})
    src = _ensure_ascending(src, "lat")

    src = src.assign_coords(lon=_normalize_lon(src["lon"]))
    src = src.sortby("lon")

    tgt_grid = _build_target_grid(glo12_path)
    regridder = xe.Regridder(src, tgt_grid, method, periodic=True, reuse_weights=False)

    if var_name:
        if var_name not in src.data_vars:
            raise ValueError(f"Variable {var_name} not found in source dataset")
        data = src[[var_name]]
    else:
        data = _select_spatial_vars(src)

    out = regridder(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path)

    src.close()


def main():
    parser = argparse.ArgumentParser(description="Regrid ERA5 to Glo12 grid")
    parser.add_argument("--era5", help="Path to a single ERA5 NetCDF")
    parser.add_argument("--era5-dir", help="Directory containing ERA5 NetCDF files")
    parser.add_argument("--pattern", default="*.nc", help="Filename glob pattern for --era5-dir")
    parser.add_argument("--glo12", required=True, help="Path to a Glo12 NetCDF (grid reference)")
    parser.add_argument("--out", help="Output NetCDF path (single file)")
    parser.add_argument("--out-dir", help="Output directory (batch mode)")
    parser.add_argument("--var", default=None, help="Variable name to regrid (default: all spatial vars)")
    parser.add_argument("--method", default="bilinear", help="Regridding method (bilinear/nearest_s2d)")
    args = parser.parse_args()

    if args.era5:
        if not args.out:
            parser.error("--out is required when using --era5")
        regrid_era5_to_glo12(args.era5, args.glo12, args.out, var_name=args.var, method=args.method)
        return

    if args.era5_dir:
        if not args.out_dir:
            parser.error("--out-dir is required when using --era5-dir")
        era5_dir = Path(args.era5_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(era5_dir.glob(args.pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {args.pattern} in {era5_dir}")
        for era5_path in files:
            out_path = out_dir / era5_path.name
            regrid_era5_to_glo12(
                str(era5_path),
                args.glo12,
                str(out_path),
                var_name=args.var,
                method=args.method,
            )
            print(f"Wrote {out_path}")
        return

    parser.error("Please provide --era5 or --era5-dir")


if __name__ == "__main__":
    main()
