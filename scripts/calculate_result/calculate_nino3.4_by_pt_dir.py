import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from nino34_utils import get_nino34_indices_and_weights, nino34_weighted_mean


def _build_grid_from_shape(shape_hw):
    h, w = shape_hw
    lat = np.linspace(-63.5, 63.5, h)
    lon = np.linspace(0.5, 359.5, w)
    return lat, lon


def _extract_lead_num(path_or_name: str):
    name = os.path.basename(path_or_name.rstrip("/\\"))
    m = re.search(r"lead_(\d+)$", name)
    return int(m.group(1)) if m else None


def _safe_name_from_relpath(rel_path: str):
    rel_path = rel_path.replace("\\", "/").strip("/")
    return rel_path.replace("/", "__")


def _load_tos_as_3d(pt_file: str):
    loaded_data = torch.load(pt_file, map_location="cpu", weights_only=False)

    if isinstance(loaded_data, np.ndarray):
        data = torch.from_numpy(loaded_data).float()
    else:
        data = loaded_data.float()

    # 统一到 (Time, Lat, Lon)
    if data.dim() == 4:
        # 常见 shape: (Time, C, Lat, Lon)
        if data.shape[1] == 1:
            data = data[:, 0]
        else:
            raise ValueError(f"Unsupported 4D tensor with C={data.shape[1]} in {pt_file}")
    elif data.dim() == 3:
        pass
    elif data.dim() == 2:
        data = data.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(data.shape)} in {pt_file}")

    return data


def _calc_nino34_for_case_dir(case_dir: str, lat_file: str, lon_file: str, var_name: str):
    pt_file = os.path.join(case_dir, "preds", f"{var_name}.pt")
    json_file = os.path.join(case_dir, "init_times.json")

    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Missing prediction file: {pt_file}")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Missing init_times.json: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        init_times = json.load(f)

    time_index = pd.to_datetime(init_times, format="%Y_%m")

    data = _load_tos_as_3d(pt_file)

    if os.path.exists(lat_file) and os.path.exists(lon_file):
        lats = np.load(lat_file).squeeze()
        lons = np.load(lon_file).squeeze()
    else:
        lats, lons = _build_grid_from_shape((data.shape[-2], data.shape[-1]))
        print(f"[WARN] Grid files not found for {case_dir}; built synthetic lat/lon from data shape.")

    lat_idx, lon_idx, weights = get_nino34_indices_and_weights(lats, lons)

    nino_raw_values = np.array(
        [
            nino34_weighted_mean(data[i].numpy(), lat_idx, lon_idx, weights)
            for i in range(data.shape[0])
        ]
    )

    if len(time_index) != len(nino_raw_values):
        raise ValueError(
            f"Length mismatch in {case_dir}: init_times={len(time_index)} vs preds={len(nino_raw_values)}"
        )

    df = pd.DataFrame(
        {
            "Date": time_index,
            "SST_Raw": nino_raw_values,
        }
    )
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Climatology"] = df.groupby("Month")["SST_Raw"].transform("mean")
    df["Nino34_Index"] = df["SST_Raw"] - df["Climatology"]

    lead_num = _extract_lead_num(case_dir)
    if lead_num is not None:
        df["Lead"] = lead_num

    return df


def main():
    parser = argparse.ArgumentParser(description="递归批量从 */preds/*.pt 计算 Nino3.4 指数")
    parser.add_argument(
        "--input_dir",
        default="../output/predict/exp2_fix",
        help="根目录，例如 ../output/predict_lead_sweep/，会递归检索所有包含 preds/<var>.pt 的目录",
    )
    parser.add_argument(
        "--output_dir",
        default="./predict/exp2_fix",
        help="输出目录：会写入每个 lead 的 CSV 和总汇总 CSV",
    )
    parser.add_argument("--var", default="tos", help="预测变量名（默认 tos，对应 preds/tos.pt）")
    parser.add_argument("--lat_file", default="../grid/lat.npy", help="纬度文件路径")
    parser.add_argument("--lon_file", default="../grid/lon.npy", help="经度文件路径")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="遇到单个目录失败时立即报错退出；默认跳过失败目录",
    )
    args = parser.parse_args()

    pt_glob = os.path.join(args.input_dir, "**", "preds", f"{args.var}.pt")
    pt_files = [p for p in glob.glob(pt_glob, recursive=True) if os.path.isfile(p)]

    case_dirs = sorted({os.path.dirname(os.path.dirname(p)) for p in pt_files})

    if len(case_dirs) == 0:
        raise FileNotFoundError(
            f"No preds/{args.var}.pt found under {args.input_dir}"
        )

    case_dirs.sort(key=lambda x: (_extract_lead_num(x) is None, _extract_lead_num(x), x))

    os.makedirs(args.output_dir, exist_ok=True)

    all_frames = []
    failed = []

    for case_dir in case_dirs:
        rel_case = os.path.relpath(case_dir, args.input_dir)
        case_name = _safe_name_from_relpath(rel_case)
        try:
            df = _calc_nino34_for_case_dir(
                case_dir=case_dir,
                lat_file=args.lat_file,
                lon_file=args.lon_file,
                var_name=args.var,
            )
            df["Case"] = rel_case.replace("\\", "/")

            per_lead_out = os.path.join(args.output_dir, f"nino34_{case_name}.csv")
            df[["Year", "Month", "SST_Raw", "Nino34_Index"]].to_csv(per_lead_out, index=False)
            print(f"[OK] {rel_case}: {len(df)} rows -> {per_lead_out}")

            all_frames.append(df)
        except Exception as e:
            failed.append((case_dir, str(e)))
            print(f"[FAIL] {rel_case}: {e}")
            if args.strict:
                raise

    if len(all_frames) == 0:
        raise RuntimeError("All lead directories failed; no output generated.")

    summary_df = pd.concat(all_frames, axis=0, ignore_index=True)

    sort_cols = [c for c in ["Case", "Lead", "Year", "Month"] if c in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    summary_out = os.path.join(args.output_dir, "nino34_all_leads.csv")
    keep_cols = ["Case", "Lead", "Year", "Month", "SST_Raw", "Nino34_Index"]
    keep_cols = [c for c in keep_cols if c in summary_df.columns]
    summary_df[keep_cols].to_csv(summary_out, index=False)

    print("-" * 40)
    print(f"Processed cases: {len(all_frames)} / {len(case_dirs)}")
    print(f"Summary saved: {os.path.abspath(summary_out)}")

    if failed:
        fail_report = os.path.join(args.output_dir, "nino34_failed_leads.txt")
        with open(fail_report, "w", encoding="utf-8") as f:
            for case_dir, err in failed:
                f.write(f"{case_dir}\t{err}\n")
        print(f"Failed cases: {len(failed)} (report: {os.path.abspath(fail_report)})")


if __name__ == "__main__":
    main()
