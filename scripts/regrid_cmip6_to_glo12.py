import argparse
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import xesmf as xe
except ImportError as exc:
    raise SystemExit("Missing dependency: xesmf. Install with `pip install xesmf`.") from exc


def _find_first_existing(names, obj):
    for name in names:
        if name in obj:
            return name
    return None


def _normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _ensure_ascending(ds, coord):
    if coord in ds.coords:
        values = ds[coord].values
        if values[0] > values[-1]:
            ds = ds.sortby(coord)
    return ds


def _build_target_grid(glo12_path):
    ds = xr.open_dataset(glo12_path, decode_times=False)
    lon_name = _find_first_existing(["longitude", "lon"], ds.coords)
    lat_name = _find_first_existing(["latitude", "lat"], ds.coords)
    depth_name = _find_first_existing(["depth", "lev", "level", "z", "deptht"], ds.coords)

    if lon_name is None or lat_name is None:
        raise ValueError("Target grid missing lon/lat coordinates")

    lon = ds[lon_name].values
    lat = ds[lat_name].values
    depth = ds[depth_name].values if depth_name else None
    ds.close()

    grid = xr.Dataset({"lon": ("lon", lon), "lat": ("lat", lat)})
    return grid, depth


def _select_singleton_dims(ds, dims):
    for dim in dims:
        if dim in ds.dims:
            ds = ds.isel({dim: 0})
    return ds


def _select_spatial_vars(ds):
    spatial_vars = []
    for name, da in ds.data_vars.items():
        if "lon" in da.dims and "lat" in da.dims:
            spatial_vars.append(name)
    if not spatial_vars:
        raise ValueError("No variables found with lon/lat dims")
    return ds[spatial_vars]


def regrid_file(cmip_path, glo12_path, output_path, var_name, method="bilinear", vertical_interp=True):
    src = xr.open_dataset(cmip_path, decode_times=False)

    lon_name = _find_first_existing(["lon", "longitude", "x"], src.coords)
    lat_name = _find_first_existing(["lat", "latitude", "y"], src.coords)
    lev_name = _find_first_existing(["lev", "depth", "level", "z", "deptht"], src.coords)

    if lon_name is None or lat_name is None:
        raise ValueError("Source dataset missing lon/lat coordinates")

    src = _select_singleton_dims(src, ["member_id", "dcpp_init_year"])
    src = src.rename({lon_name: "lon", lat_name: "lat"})
    src = _ensure_ascending(src, "lat")
    src = src.assign_coords(lon=_normalize_lon(src["lon"]))
    src = src.sortby("lon")

    tgt_grid, tgt_depth = _build_target_grid(glo12_path)

    if var_name:
        if var_name not in src.data_vars:
            raise ValueError(f"Variable {var_name} not found in source dataset")
        data = src[[var_name]]
    else:
        data = _select_spatial_vars(src)

    regridder = xe.Regridder(data, tgt_grid, method, periodic=True, reuse_weights=False)
    out = regridder(data)

    if vertical_interp and lev_name and tgt_depth is not None:
        out = out.rename({lev_name: "lev"}) if lev_name in out.coords else out
        out = out.interp(lev=tgt_depth)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path)

    src.close()


def main():
    parser = argparse.ArgumentParser(description="Regrid CMIP6 split files to Glo12 grid")
    parser.add_argument("--cmip-dir", required=True, help="Directory containing CMIP6 split NetCDF files")
    parser.add_argument("--glo12", required=True, help="Path to a Glo12 NetCDF (grid reference)")
    parser.add_argument("--var", default=None, help="Variable name to regrid (default: all spatial vars)")
    parser.add_argument("--out-dir", required=True, help="Output directory for regridded NetCDFs")
    parser.add_argument("--pattern", default="part_*.nc", help="Filename glob pattern")
    parser.add_argument("--method", default="bilinear", help="Regridding method (bilinear/nearest_s2d)")
    parser.add_argument("--no-vertical", action="store_true", help="Disable vertical interpolation")
    args = parser.parse_args()

    cmip_dir = Path(args.cmip_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cmip_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {cmip_dir}")

    for cmip_path in files:
        out_path = out_dir / cmip_path.name
        regrid_file(
            str(cmip_path),
            args.glo12,
            str(out_path),
            var_name=args.var,
            method=args.method,
            vertical_interp=not args.no_vertical,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
