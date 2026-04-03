import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt



VERTICAL_DIM_CANDIDATES = {
    "lev", "level", "depth", "deptht", "z", "st_ocean", "olevel",
}


def run_cmd(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ CDO command failed:")
        print(" ".join(args))
        print("---- STDOUT ----")
        print(result.stdout)
        print("---- STDERR ----")
        print(result.stderr)
        raise RuntimeError("CDO failed")


def find_nc_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    files = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted([fp for fp in files if fp.is_file()])


def has_vertical_levels(ds: xr.Dataset) -> bool:
    for var in ds.data_vars:
        dims = {d.lower() for d in ds[var].dims}
        if dims.intersection(VERTICAL_DIM_CANDIDATES):
            return True
    return False


def parse_yyyymm_from_name(path: Path) -> str | None:
    m = re.search(r"(\d{6})\.nc$", path.name)
    return m.group(1) if m else None


def month_key_to_out_name(yyyymm: str) -> str:
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    return f"{year}_{month}.nc"


# =========================
# ⭐ 关键函数：插值 + 去 NaN
# =========================
def postprocess_dataset(ds: xr.Dataset) -> xr.Dataset:
    # 1️⃣ 时间插值
    if "time" in ds.dims:
        ds = ds.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

    # 2️⃣ 空间插值
    if "lat" in ds.dims:
        ds = ds.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    if "lon" in ds.dims:
        ds = ds.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")

    # 3️⃣ 强制转换为浮点类型，避免 isnan 报错
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            ds[var] = ds[var].astype("float64")

    # 4️⃣ 最终兜底
    ds = ds.fillna(0)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/workspace/lutianyou/program/download/CAS-ESM2-0/all")
    parser.add_argument("--output-dir", default="../../data/process/CAS-ESM2-0")
    parser.add_argument("--grid", default="../grid")
    parser.add_argument("--zaxis", default="../zaxis.txt")
    parser.add_argument("--levels", default="10,15,30,50,75,100,125,150,200,250,300,400,500,600,800,1000")
    parser.add_argument("--pattern", default="*.nc")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if shutil.which("cdo") is None:
        raise RuntimeError("CDO not found")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    grid_path = Path(args.grid)
    zaxis_path = Path(args.zaxis)

    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = find_nc_files(input_dir, args.pattern, args.recursive)
    if not nc_files:
        raise RuntimeError("No nc files found")

    temp_root = Path(tempfile.mkdtemp(prefix="nc_preproc_", dir="./"))

    try:
        split_files_by_month: dict[str, list[Path]] = {}

        # =========================
        # Step 1: CDO处理
        # =========================
        for i, src_path in enumerate(nc_files):
            print(f"[{i+1}/{len(nc_files)}] {src_path}")

            with xr.open_dataset(src_path) as ds:
                if "expver" in ds.dims:
                    print("  removing expver")
                    ds = ds.isel(expver=0)

                if "number" in ds.dims:
                    print("  removing number")
                    ds = ds.isel(number=0)
                is_3d = has_vertical_levels(ds)

            remap_path = temp_root / f"{src_path.stem}_remap.nc"
            run_cmd(["cdo", "-b", "f64", f"remapbil,{grid_path}", str(src_path), str(remap_path)])

            if is_3d:
                int_path = temp_root / f"{src_path.stem}_int.nc"
                processed_path = temp_root / f"{src_path.stem}_proc.nc"
                run_cmd(["cdo", f"intlevel,{args.levels}", str(remap_path), str(int_path)])
                run_cmd(["cdo", f"setzaxis,{zaxis_path}", str(int_path), str(processed_path)])
            else:
                processed_path = remap_path

            # ❗不做 setmisstoc

            split_prefix = temp_root / f"{src_path.stem}_ym_"
            run_cmd(["cdo", "splityearmon", str(processed_path), str(split_prefix)])
            # --- 及时释放空间 ---
            if remap_path.exists():
                remap_path.unlink()  # 删除 remap 文件
            if is_3d and int_path.exists():
                int_path.unlink()    # 删除插值后的中间文件
            # -----------------------

            for part_path in sorted(temp_root.glob(f"{src_path.stem}_ym_*.nc")):
                key = parse_yyyymm_from_name(part_path)
                if key:
                    split_files_by_month.setdefault(key, []).append(part_path)

        # =========================
        # Step 2: merge + 插值
        # =========================
        for key in sorted(split_files_by_month):
            out_path = output_dir / month_key_to_out_name(key)
            parts = split_files_by_month[key]

            if len(parts) == 1:
                ds = xr.open_dataset(parts[0])
            else:
                # 取代 cdo merge，使用 xarray 合并
                datasets = []
                for p in parts:
                    temp_ds = xr.open_dataset(p)
                    # 统一时间坐标，防止因毫秒级差异无法 merge
                    if "time" in temp_ds.coords:
                        temp_ds['time'] = [np.datetime64(f"{key[:4]}-{key[4:6]}-01")]
                    datasets.append(temp_ds)
                
                # combine_by_coords 会自动处理 2D/3D 变量的合并
                ds = xr.merge(datasets, compat='override') 

            # =========================
            # ⭐ Python后处理
            # =========================
            # 处理变量名（如果是 ERA5 数据常见的 valid_time）
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            elif "valid_time" in ds.coords:
                ds = ds.rename({"valid_time": "time"})

            ds = postprocess_dataset(ds)

            # 检查 NaN 并保存
            total_nan = 0
            for var in ds.data_vars:
                if np.issubdtype(ds[var].dtype, np.floating):
                    total_nan += np.isnan(ds[var].values).sum()
            print(f"{out_path.name} | remaining NaN: {total_nan}")
            ds.to_netcdf(out_path)

            # mask = np.isnan(ds['zos'][0].values)

            # plt.imshow(mask)
            # plt.colorbar()
            # plt.title("NaN mask")
            # plt.show()
        print("Done!")
    finally:
        print(f"Cleaning temp dir: {temp_root}")
        shutil.rmtree(temp_root, ignore_errors=True)

    


if __name__ == "__main__":
    main()