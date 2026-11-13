"""Build point-in-time (truncated) data trees for FNR retraining.

Every month file up to and including ``--cutoff`` (YYYYMM, must be a month end)
is hard-linked into the output directory. Missing later months are skipped by
``champion.fnr.training.base_training.read_public_tables``.

Usage:
  python scripts/fnr_make_cutoff_data.py --cutoff 202112 --out _data_trunc/cut2021_12
"""
import argparse
import os
from pathlib import Path

TABLES = ("MarketData", "Factor33_winsor", "Barra_factor", "MarketDataRawOpen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data_temp/Data"))
    ap.add_argument("--cutoff", type=int, required=True, help="YYYYMM, inclusive")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cy, cm = args.cutoff // 100, args.cutoff % 100
    made = 0
    for table in TABLES:
        src_root = args.src / table
        if not src_root.is_dir():
            continue
        for year in range(2016, 2023):
            for month in range(1, 13):
                if (year, month) > (cy, cm):
                    continue
                src = src_root / str(year) / f"{month:02d}" / f"{year}{month:02d}.parquet"
                if not src.exists():
                    continue
                dst = args.out / table / str(year) / f"{month:02d}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    continue
                os.link(src, dst)
                made += 1
    print(f"linked {made} files -> {args.out} (cutoff {args.cutoff})")


if __name__ == "__main__":
    main()
