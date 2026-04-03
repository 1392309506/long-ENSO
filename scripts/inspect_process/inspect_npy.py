import argparse
from pathlib import Path

import numpy as np


LON_CANDIDATES = ["lon", "longitude", "x", "nav_lon"]
LAT_CANDIDATES = ["lat", "latitude", "y", "nav_lat"]


def _format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _infer_resolution(values):
    values = np.asarray(values).reshape(-1)
    if values.size < 2:
        return None
    diffs = np.diff(values)
    if not np.allclose(diffs, diffs[0]):
        return None
    return float(diffs[0])


def _candidate_variable_name(file_path, root_dir=None):
    file_path = Path(file_path)
    stem = file_path.stem
    if stem:
        name = stem.split("_")[0]
        if name and name.lower() not in LON_CANDIDATES + LAT_CANDIDATES:
            return name

    parent = file_path.parent.name
    if parent and parent.lower() not in LON_CANDIDATES + LAT_CANDIDATES:
        if root_dir is None:
            return parent
        root_dir = Path(root_dir)
        try:
            rel_parent = file_path.parent.relative_to(root_dir)
            if len(rel_parent.parts) >= 1:
                return rel_parent.parts[0]
        except ValueError:
            return parent
    return None


def _find_coord_file(search_dir, candidate_names):
    lower_names = {f"{name}.npy" for name in candidate_names}
    for p in search_dir.glob("*.npy"):
        if p.name.lower() in lower_names:
            return p
    return None


def _find_lon_lat_files(file_path, root_dir=None):
    file_path = Path(file_path)
    search_dirs = [file_path.parent]
    if root_dir is not None:
        root_dir = Path(root_dir)
        if root_dir.exists() and root_dir not in search_dirs:
            search_dirs.append(root_dir)
        for parent in file_path.parents:
            if parent in search_dirs:
                continue
            search_dirs.append(parent)
            if root_dir is not None and parent == root_dir:
                break

    lon_file = None
    lat_file = None
    for d in search_dirs:
        if lon_file is None:
            lon_file = _find_coord_file(d, LON_CANDIDATES)
        if lat_file is None:
            lat_file = _find_coord_file(d, LAT_CANDIDATES)
        if lon_file is not None and lat_file is not None:
            break
    return lon_file, lat_file


def _print_geo_info(npy_file, root_dir=None):
    lon_file, lat_file = _find_lon_lat_files(npy_file, root_dir=root_dir)
    if lon_file is None or lat_file is None:
        print("  geo: lon/lat coordinate files not found")
        return

    lon = np.load(lon_file, allow_pickle=False)
    lat = np.load(lat_file, allow_pickle=False)
    lon_flat = np.asarray(lon).reshape(-1)
    lat_flat = np.asarray(lat).reshape(-1)

    if lon_flat.size == 0 or lat_flat.size == 0:
        print("  geo: coordinate file is empty")
        return

    lon_res = _infer_resolution(lon_flat)
    lat_res = _infer_resolution(lat_flat)
    print("  lon file:", str(lon_file))
    print("  lat file:", str(lat_file))
    print("  lon range:", float(np.nanmin(lon_flat)), "to", float(np.nanmax(lon_flat)), "size:", lon_flat.size)
    print("  lat range:", float(np.nanmin(lat_flat)), "to", float(np.nanmax(lat_flat)), "size:", lat_flat.size)
    print("  lon res:", lon_res, "lat res:", lat_res)


def _print_array_stats(arr):
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    print("  ndim:", arr.ndim)
    print("  size:", arr.size)
    print("  nbytes:", arr.nbytes, f"({_format_bytes(arr.nbytes)})")

    if arr.size == 0:
        print("  stats: empty array")
        return

    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        arr_mean = np.nanmean(arr)
        print("  min:", float(arr_min))
        print("  max:", float(arr_max))
        print("  mean:", float(arr_mean))
    else:
        print("  stats: skipped (non-numeric dtype)")


def inspect_single_npy(npy_path, root_dir=None):
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(str(npy_path))
    if npy_path.suffix.lower() != ".npy":
        raise ValueError(f"Not a .npy file: {npy_path}")

    print("NPY file:", str(npy_path))
    variable_name = _candidate_variable_name(npy_path, root_dir=root_dir)
    if variable_name:
        print("Variable (inferred):", variable_name)
    array = np.load(npy_path, allow_pickle=False)
    _print_array_stats(array)
    _print_geo_info(npy_path, root_dir=root_dir)


def inspect_npy_path(path):
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))

    if target.is_file():
        inspect_single_npy(target, root_dir=target.parent)
        return

    npy_files = sorted(target.rglob("*.npy"))
    print("NPY dir:", str(target))
    print("Total .npy files:", len(npy_files))

    if not npy_files:
        print("No .npy files found")
        return

    for npy_file in npy_files:
        print("-" * 60)
        rel_path = npy_file.relative_to(target)
        print("File:", str(rel_path))
        variable_name = _candidate_variable_name(npy_file, root_dir=target)
        if variable_name:
            print("Variable (inferred):", variable_name)
        array = np.load(npy_file, allow_pickle=False)
        _print_array_stats(array)
        _print_geo_info(npy_file, root_dir=target)


def main():
    parser = argparse.ArgumentParser(description="Inspect .npy file(s) and print shape/type/statistics")
    parser.add_argument("--path", default = "../data/godas/tos/1980_1.npy", type=str, help="Path to a .npy file or a directory")
    args = parser.parse_args()

    inspect_npy_path(args.path)


if __name__ == "__main__":
    main()
