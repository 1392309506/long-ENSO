import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


def find_nc_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted([fp for fp in files if fp.is_file()])


def clean_dataset_nonfinite(ds: xr.Dataset, fill_value: float) -> tuple[xr.Dataset, dict[str, int]]:
    cleaned = ds.copy(deep=True)
    changed_stats: dict[str, int] = {}

    for var in cleaned.data_vars:
        da = cleaned[var]
        if not np.issubdtype(da.dtype, np.number):
            continue

        values = da.values
        if np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.complexfloating):
            bad_mask = ~np.isfinite(values)
            bad_count = int(np.count_nonzero(bad_mask))
            if bad_count > 0:
                values = np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)
                cleaned[var].values = values
            changed_stats[var] = bad_count
        else:
            changed_stats[var] = 0

    return cleaned, changed_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean existing NetCDF files by replacing NaN/Inf/missing numeric values."
        )
    )
    parser.add_argument("--input-dir", default="../../download/era5", help="Directory containing existing .nc files")
    parser.add_argument("--output-dir", default="../data/process/era5", help="Output directory for cleaned files. If omitted and --in-place is set, overwrite source files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files in place. If set, --output-dir is ignored.",
    )
    parser.add_argument("--pattern", default="*.nc", help="File pattern to match nc files (default: *.nc)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search sub-directories")
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Value used to replace NaN/Inf/missing entries (default: 0.0)",
    )
    parser.add_argument(
        "--on-exist",
        default="overwrite",
        choices=["overwrite", "skip", "error"],
        help="Behavior when output file exists in non in-place mode (default: overwrite)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input directory: {input_dir}")

    if not args.in_place and args.output_dir is None:
        raise ValueError("Please provide --output-dir, or use --in-place.")

    output_dir = None if args.in_place else Path(args.output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = find_nc_files(input_dir, args.pattern, args.recursive)
    if not nc_files:
        raise FileNotFoundError(f"No nc files found: dir={input_dir}, pattern={args.pattern}")

    processed_files = 0
    skipped_files = 0
    total_replaced = 0

    for src_path in nc_files:
        rel_path = src_path.relative_to(input_dir)
        if args.in_place:
            dst_path = src_path
        else:
            dst_path = output_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                if args.on_exist == "skip":
                    skipped_files += 1
                    continue
                if args.on_exist == "error":
                    raise FileExistsError(f"Output already exists: {dst_path}")

        with xr.open_dataset(src_path, decode_times=False) as ds:
            cleaned_ds, stats = clean_dataset_nonfinite(ds, args.fill_value)

        replaced_in_file = sum(stats.values())
        total_replaced += replaced_in_file

        if args.in_place:
            tmp_path = src_path.with_suffix(src_path.suffix + ".tmp")
            cleaned_ds.to_netcdf(tmp_path)
            shutil.move(str(tmp_path), str(src_path))
        else:
            cleaned_ds.to_netcdf(dst_path)

        processed_files += 1
        print(f"[{processed_files}/{len(nc_files)}] {src_path.name} replaced={replaced_in_file}")

    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Total replaced non-finite values: {total_replaced}")
    if args.in_place:
        print(f"Mode: in-place, source directory updated: {input_dir.resolve()}")
    else:
        print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()