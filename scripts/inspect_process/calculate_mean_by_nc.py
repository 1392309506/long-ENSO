#!/usr/bin/env python
import argparse
import json
from typing import Dict, Any

import numpy as np
import xarray as xr


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def compute_means(ds: xr.Dataset, var_name: str = None) -> Dict[str, float]:
    means: Dict[str, float] = {}

    if var_name is not None:
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found. Available vars: {list(ds.data_vars)}"
            )
        da = ds[var_name]
        means[var_name] = float(_to_python_scalar(da.mean(skipna=True).values))
        return means

    for name, da in ds.data_vars.items():
        means[name] = float(_to_python_scalar(da.mean(skipna=True).values))

    return means


def main():
    parser = argparse.ArgumentParser(description="Compute variable mean(s) from one NetCDF file.")
    parser.add_argument("nc_path",default=None, help="Path to a .nc file")
    parser.add_argument(
        "--var",
        default=None,
        help="Variable name to compute mean for. If omitted, compute all data variables.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="xarray engine, e.g. netcdf4/h5netcdf/scipy. If omitted, xarray auto-detects.",
    )
    parser.add_argument(
        "--print-overall",
        action="store_true",
        help="Also print one overall mean that averages all variable means.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results in JSON format.",
    )
    args = parser.parse_args()

    open_kwargs = {}
    if args.engine:
        open_kwargs["engine"] = args.engine

    ds = xr.open_dataset(args.nc_path, **open_kwargs)
    try:
        means = compute_means(ds, var_name=args.var)
    finally:
        ds.close()

    if args.print_overall and len(means) > 0:
        overall = float(np.mean(list(means.values())))
        means["overall_mean_of_variables"] = overall

    if args.json:
        print(json.dumps(means, indent=2, ensure_ascii=False))
        return

    print("=== Variable Mean(s) ===")
    for k in sorted(means.keys()):
        print(f"{k}: {means[k]:.10f}")


if __name__ == "__main__":
    main()
