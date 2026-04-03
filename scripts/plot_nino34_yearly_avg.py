import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def compute_yearly_mean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"Year", "Nino34_Index"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {missing}")

    yearly = (
        df.groupby("Year", as_index=False)["Nino34_Index"]
        .mean()
        .rename(columns={"Nino34_Index": "Nino34_YearlyMean"})
    )
    return yearly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算两个 CSV 的 Nino3.4 指数年平均并绘图"
    )
    parser.add_argument(
        "--csv1",
        type=Path,
        default=Path("./nino3.4_fix/exp1_fix_2010-2021.csv"),
        help="第一个 CSV 路径",
    )
    parser.add_argument(
        "--csv2",
        type=Path,
        default=Path("./nino3.4_fix/nino34_from_nc_dir.csv"),
        help="第二个 CSV 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./nino3.4_fix"),
        help="输出目录（年均 CSV 与图片）",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    yearly1 = compute_yearly_mean(args.csv1)
    yearly2 = compute_yearly_mean(args.csv2)

    out_csv1 = args.out_dir / f"{args.csv1.stem}_yearly_mean.csv"
    out_csv2 = args.out_dir / f"{args.csv2.stem}_yearly_mean.csv"
    yearly1.to_csv(out_csv1, index=False)
    yearly2.to_csv(out_csv2, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(yearly1["Year"], yearly1["Nino34_YearlyMean"], marker="o", label=args.csv1.stem)
    plt.plot(yearly2["Year"], yearly2["Nino34_YearlyMean"], marker="o", label=args.csv2.stem)
    plt.xlabel("Year")
    plt.ylabel("Nino3.4")
    plt.title("Yearly Mean Nino3.4 Index")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_png = args.out_dir / "nino34_yearly_mean_compare.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"已保存: {out_csv1}")
    print(f"已保存: {out_csv2}")
    print(f"已保存: {out_png}")


if __name__ == "__main__":
    main()
