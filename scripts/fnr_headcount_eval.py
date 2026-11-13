"""Sweep the protected-head size (RH{head_count}) of the FNR output layer.

`merge_scores` keeps the top ``head_count`` names of the reference signal and
reorders the rest with the K3 signal. The champion fixes 60; we re-select it on
the main OOS window and verify on the sub window.

Usage:
  python scripts/fnr_headcount_eval.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from champion.fnr.inference.rank_merge import merge_scores  # noqa: E402
from dual_window_validate import _df, daily_rankic, ir_backtest  # noqa: E402

WINDOWS = [
    ("main", 20220101, 20221231),
    ("sub", 20210701, 20211231),
]


def build_scores(internal: pd.DataFrame, head_count: int) -> pd.DataFrame:
    parts = []
    for _, day in internal.groupby("日期", sort=True):
        rh = day[["日期", "代码", "rh50"]].rename(columns={"rh50": "score"})
        rh = rh.assign(date=rh["日期"].astype(str), code=rh["代码"])
        k3 = day[["日期", "代码", "k3"]].rename(columns={"k3": "score"})
        k3 = k3.assign(date=k3["日期"].astype(str), code=k3["代码"])
        merged = merge_scores(rh[["date", "code", "score"]], k3[["date", "code", "score"]], head_count)
        merged["日期"] = day["日期"].iloc[0]
        merged = merged.rename(columns={"code": "代码"})
        parts.append(merged[["日期", "代码", "score"]])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="20,30,40,50,60,70,80,100,120")
    ap.add_argument("--ir-head", default="60,", help="comma separated head_counts to run through IR")
    args = ap.parse_args()
    grid = [int(x) for x in args.grid.split(",") if x]

    results = {}
    for tag, start, end in WINDOWS:
        internal_path = ROOT / f"result/fnr_internal_{tag}.parquet"
        if not internal_path.exists():
            print(f"skip {tag}: {internal_path} missing")
            continue
        internal = pd.read_parquet(internal_path)
        base = _df.loc[_df["日期"].between(start, end)]
        rows = {}
        for hc in grid:
            scores = build_scores(internal, hc)
            seg = base.merge(scores, on=["日期", "代码"], how="left", validate="one_to_one")
            ric = daily_rankic(seg, "score")
            rows[hc] = (ric["mean_rankic"], ric["rankicir"], ric["positive_ratio"])
        results[tag] = rows

    for tag, rows in results.items():
        print(f"\n[{tag}] head_count  RankIC   ICIR   正IC%")
        for hc, (ic, icir, pos) in rows.items():
            print(f"         {hc:<10} {ic:+.4f}  {icir:.3f}  {pos:.1%}")

    if "main" not in results:
        return
    best = max(results["main"], key=lambda h: results["main"][h][0])
    print(f"\nbest head_count on main = {best}")
    if "sub" in results:
        ic, icir, pos = results["sub"][best]
        print(f"[sub] head_count={best} RankIC={ic:+.4f} ICIR={icir:.3f} 正IC%={pos:.1%}")

    for hc in [int(x) for x in args.ir_head.split(",") if x]:
        for tag, start, end in WINDOWS:
            internal_path = ROOT / f"result/fnr_internal_{tag}.parquet"
            if not internal_path.exists():
                continue
            internal = pd.read_parquet(internal_path)
            scores = build_scores(internal, hc)
            seg = _df.loc[_df["日期"].between(start, end)].merge(
                scores, on=["日期", "代码"], how="left", validate="one_to_one")
            ir = ir_backtest(seg, "score", str(pd.Timestamp(str(start)).date()),
                             str(pd.Timestamp(str(end)).date()))
            print(f"[{tag}] head_count={hc} IR={ir.get('information_ratio')} "
                  f"TE={ir.get('tracking_error')} 超额={ir.get('annualized_excess_return')}")


if __name__ == "__main__":
    main()
