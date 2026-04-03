import xarray as xr
import numpy as np
from pathlib import Path


def check_file(fp: Path):
    print(f"\n===== Checking: {fp.name} =====")

    ds = xr.open_dataset(fp)
    data = ds.to_array().values  # [var, time?, lat, lon]

    # =========================
    # 1. NaN 检查
    # =========================
    nan_count = np.isnan(data).sum()
    print(f"NaN count: {nan_count}")

    # =========================
    # 2. Inf 检查
    # =========================
    inf_count = np.isinf(data).sum()
    print(f"Inf count: {inf_count}")

    # =========================
    # 3. 基本统计
    # =========================
    print(f"Min: {np.nanmin(data):.4f}")
    print(f"Max: {np.nanmax(data):.4f}")
    print(f"Mean: {np.nanmean(data):.4f}")
    print(f"Std: {np.nanstd(data):.4f}")

    # =========================
    # 4. 全0比例（非常关键）
    # =========================
    zero_ratio = (data == 0).sum() / data.size
    print(f"Zero ratio: {zero_ratio:.4%}")

    # =========================
    # 5. 每变量检查
    # =========================
    for var in ds.data_vars:
        arr = ds[var].values

        nan = np.isnan(arr).sum()
        zero = (arr == 0).sum()

        print(f"\n[var: {var}]")
        print(f"  shape: {arr.shape}")
        print(f"  NaN: {nan}")
        print(f"  Zero ratio: {zero / arr.size:.4%}")
        print(f"  Min: {np.nanmin(arr):.4f}, Max: {np.nanmax(arr):.4f}")

    # =========================
    # 6. 时间维检查
    # =========================
    if "time" in ds.dims:
        print(f"\nTime steps: {ds.dims['time']}")
        try:
            print(f"Time range: {ds['time'].values[0]} -> {ds['time'].values[-1]}")
        except:
            print("Time decode failed")

    ds.close()


def check_directory(data_dir):
    files = sorted(Path(data_dir).glob("*.nc"))
    for fp in files:
        check_file(fp)


if __name__ == "__main__":
    check_directory("../../data/process/era5")