import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


def open_dataset_with_cftime(nc_path: Path) -> xr.Dataset:
    # xarray>=2025 prefers CFDatetimeCoder over use_cftime kwarg.
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(nc_path, decode_times=time_coder)
    except AttributeError:
        return xr.open_dataset(nc_path, decode_times=True, use_cftime=True)
        
def find_nc_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(input_dir.rglob(pattern))
    else:
        files = sorted(input_dir.glob(pattern))
    return [fp for fp in files if fp.is_file()]


def parse_vars(vars_arg: str | None) -> set[str] | None:
    if vars_arg is None:
        return None
    names = [x.strip() for x in vars_arg.split(",") if x.strip()]
    return set(names) if names else None


def iter_export_vars(ds: xr.Dataset, selected_vars: set[str] | None) -> Iterable[str]:
    for var in ds.data_vars:
        if selected_vars is not None and var not in selected_vars:
            continue
        yield var


def ensure_time_coord(ds: xr.Dataset, time_dim: str) -> xr.DataArray:
    if time_dim in ds.coords:
        return ds[time_dim]
    if time_dim in ds.dims:
        return ds[time_dim]
    raise ValueError(f"Time coordinate or dimension '{time_dim}' not found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert all NetCDF files in a directory to per-month NPY files grouped by variable."
    )
    parser.add_argument("--input-dir", default="../../data/process/BCC-CSM2-MR/thetao", help="Directory containing .nc files")
    parser.add_argument("--output-dir", default="../../data/train_data/BCC-CSM2-MR", help="Output root directory")
    parser.add_argument(
        "--pattern",
        default="*.nc",
        help="File pattern to match NetCDF files (default: *.nc)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search NetCDF files in sub-directories",
    )
    parser.add_argument(
        "--time-dim",
        default="time",
        help="Time coordinate/dimension name (default: time)",
    )
    parser.add_argument(
        "--vars",
        default=None,
        help="Comma-separated variable names to export. Default exports all data variables with time dimension.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Output dtype (default: float32)",
    )
    parser.add_argument(
        "--on-exist",
        default="overwrite",
        choices=["overwrite", "skip", "error"],
        help="Behavior when output file already exists (default: overwrite)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory is invalid: {input_dir}")

    selected_vars = parse_vars(args.vars)
    nc_files = find_nc_files(input_dir, args.pattern, args.recursive)
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found under {input_dir} with pattern '{args.pattern}'.")

    total_written = 0
    total_skipped = 0
    total_files = 0
    total_non_numeric_skipped = 0

    for nc_path in nc_files:
        total_files += 1
        with open_dataset_with_cftime(nc_path) as ds:
            time_coord = ensure_time_coord(ds, args.time_dim)

            years = time_coord.dt.year.values
            months = time_coord.dt.month.values

            for var in iter_export_vars(ds, selected_vars):
                da = ds[var]
                if args.time_dim not in da.dims:
                    continue
                if not np.issubdtype(da.dtype, np.number):
                    print(f"[Skip non-numeric var] file={nc_path.name} var={var} dtype={da.dtype}")
                    total_non_numeric_skipped += 1
                    continue

                var_out_dir = output_dir / var
                var_out_dir.mkdir(parents=True, exist_ok=True)

                for i in range(da.sizes[args.time_dim]):
                    year = int(years[i])
                    month = int(months[i])
                    out_path = var_out_dir / f"{year}_{month}.npy"

                    if out_path.exists():
                        if args.on_exist == "skip":
                            total_skipped += 1
                            continue
                        if args.on_exist == "error":
                            raise FileExistsError(f"Output already exists: {out_path}")

                    arr = da.isel({args.time_dim: i}).values
                    arr = np.asarray(arr, dtype=args.dtype)
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    np.save(out_path, arr)
                    total_written += 1

    print(f"Processed NC files: {total_files}")
    print(f"Saved NPY files: {total_written}")
    print(f"Skipped existing: {total_skipped}")
    print(f"Skipped non-numeric vars: {total_non_numeric_skipped}")
    print(f"Output root: {os.path.abspath(str(output_dir))}")


if __name__ == "__main__":
    main()