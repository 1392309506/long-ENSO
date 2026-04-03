import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from nino34_utils import get_nino34_indices_and_weights, nino34_weighted_mean


def parse_year_month_from_name(file_name: str):
    name = os.path.splitext(os.path.basename(file_name))[0]
    m = re.search(r"(\d{4})_(\d{1,2})$", name)
    if m is None:
        return None
    year = int(m.group(1))
    month = int(m.group(2))
    if month < 1 or month > 12:
        return None
    return year, month


def detect_coord_name(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name
    return None


def detect_data_var(ds: xr.Dataset, prefer_var: str | None):
    if prefer_var and prefer_var in ds.data_vars:
        return prefer_var

    for name in ["sst", "tos", "SST", "TOS"]:
        if name in ds.data_vars:
            return name

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(
        f"无法自动识别海温变量，请通过 --var 指定。可选变量: {list(ds.data_vars)}"
    )


def to_2d_field(da: xr.DataArray, lat_name: str, lon_name: str):
    keep_dims = {lat_name, lon_name}
    for dim in da.dims:
        if dim not in keep_dims:
            da = da.isel({dim: 0})

    if da.ndim != 2:
        raise ValueError(f"变量无法转换为2D场，当前维度: {da.dims}, 形状: {da.shape}")

    if da.dims != (lat_name, lon_name):
        da = da.transpose(lat_name, lon_name)

    return da.values


def main():
    parser = argparse.ArgumentParser(description="从 nc_dir 批量计算 Nino3.4 指数")
    parser.add_argument("--nc_dir", default="../output/predict/exp2_fix", help="包含月度 nc 文件的目录")
    parser.add_argument("--output", default="./nino3.4/exp2_from_nc_dir.csv", help="输出 CSV 路径")
    parser.add_argument("--var", default=None, help="海温变量名（如 sst/tos），默认自动识别")
    parser.add_argument("--pattern", default="*.nc", help="文件匹配模式，默认 *.nc")
    parser.add_argument("--strict", action="store_true", help="遇到坏文件时报错退出（默认跳过坏文件）")
    args = parser.parse_args()

    file_list = glob.glob(os.path.join(args.nc_dir, args.pattern))
    if len(file_list) == 0:
        raise FileNotFoundError(f"目录中未找到文件: {args.nc_dir} / {args.pattern}")

    parsed = []
    for fp in file_list:
        ym = parse_year_month_from_name(fp)
        if ym is not None:
            parsed.append((fp, ym[0], ym[1]))

    if len(parsed) == 0:
        raise ValueError("未在文件名中解析到年月，期望格式如 glo12v1_1993_01.nc")

    parsed.sort(key=lambda x: (x[1], x[2]))

    lat_name = None
    lon_name = None
    var_name = None
    lat_grid = None
    lon_grid = None
    skipped_files = []

    for fp, _, _ in parsed:
        try:
            with xr.open_dataset(fp) as ds0:
                lat_name = detect_coord_name(ds0, ["lat", "latitude", "nav_lat", "y"])
                lon_name = detect_coord_name(ds0, ["lon", "longitude", "nav_lon", "x"])

                if lat_name is None or lon_name is None:
                    raise ValueError(
                        f"无法识别经纬度坐标名。coords={list(ds0.coords)} dims={list(ds0.dims)}"
                    )

                var_name = detect_data_var(ds0, args.var)
                lat_grid = ds0[lat_name].values
                lon_grid = ds0[lon_name].values
                break
        except Exception as e:
            skipped_files.append((fp, str(e)))
            if args.strict:
                raise

    if lat_name is None or lon_name is None or var_name is None or lat_grid is None or lon_grid is None:
        raise RuntimeError("没有可读取的 nc 文件可用于初始化坐标和变量，请检查数据完整性。")

    lat_idx, lon_idx, weights = get_nino34_indices_and_weights(lat_grid, lon_grid)

    results = []
    for fp, year, month in parsed:
        try:
            with xr.open_dataset(fp) as ds:
                if var_name not in ds.data_vars:
                    var_name = detect_data_var(ds, args.var)

                field2d = to_2d_field(ds[var_name], lat_name, lon_name)
                mean_val = nino34_weighted_mean(field2d, lat_idx, lon_idx, weights)

            results.append(
                {
                    "Year": year,
                    "Month": month,
                    "SST_Raw": mean_val,
                }
            )
        except Exception as e:
            skipped_files.append((fp, str(e)))
            if args.strict:
                raise
            print(f"[跳过坏文件] {fp} :: {e}")

    if len(results) == 0:
        raise RuntimeError("所有文件都读取失败，无法生成结果。")

    df = pd.DataFrame(results)
    df["Nino34_Index"] = df["SST_Raw"] - df.groupby("Month")["SST_Raw"].transform("mean")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(args.output, index=False)

    print(f"处理完成，共 {len(df)} 条记录。")
    print(f"跳过坏文件: {len(skipped_files)} 个")
    if skipped_files:
        report_path = os.path.splitext(args.output)[0] + "_file_check_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            for fp, err in skipped_files:
                f.write(f"{fp}\t{err}\n")
        print(f"坏文件报告已保存: {os.path.abspath(report_path)}")
    print(f"结果已保存至: {os.path.abspath(args.output)}")
    print(df.head(12))


if __name__ == "__main__":
    main()
