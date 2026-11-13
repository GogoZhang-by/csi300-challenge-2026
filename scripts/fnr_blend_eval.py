"""Blend FNR scores with the 71-feature LightGBM scores and evaluate honestly.

Per-day equal-weight rank blend:
    score = (1 - w) * rank_pct(fnr) + w * rank_pct(baseline)

Weight is selected on the main window (2022) and then checked on the sub window
(2021H2). Nothing is tuned on the sub window.

Usage:
  python scripts/fnr_blend_eval.py --fnr-main result/fnr_scores_main.parquet \
      --fnr-sub result/fnr_scores_sub.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dual_window_validate import _df, daily_rankic, ir_backtest  # noqa: E402


def load_scores(path: Path, col: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if col and col not in frame.columns:
        frame = frame.rename(columns={frame.columns[-1]: col})
    return frame


def blend(seg: pd.DataFrame, w: float) -> pd.DataFrame:
    out = seg.copy()
    a = out.groupby("日期")["fnr"].rank(pct=True)
    b = out.groupby("日期")["score71"].rank(pct=True)
    out["score"] = (1 - w) * a + w * b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnr-main", type=Path, default=ROOT / "result/fnr_scores_main.parquet")
    ap.add_argument("--fnr-sub", type=Path, default=ROOT / "result/fnr_scores_sub.parquet")
    ap.add_argument("--grid", default="0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--ir", default="")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    windows = {}
    for tag, fnr_path, start, end in [
        ("main", args.fnr_main, 20220101, 20221231),
        ("sub", args.fnr_sub, 20210701, 20211231),
    ]:
        fnr = load_scores(fnr_path).rename(columns={"score": "fnr"})
        base = load_scores(ROOT / f"result/baseline_scores_{tag}.parquet")
        seg = _df.loc[_df["日期"].between(start, end)].merge(
            fnr[["日期", "代码", "fnr"]], on=["日期", "代码"], how="left", validate="one_to_one"
        ).merge(base, on=["日期", "代码"], how="left", validate="one_to_one")
        windows[tag] = (seg, start, end)

    print("w      main_IC   main_ICIR   sub_IC    sub_ICIR")
    best = None
    for w in grid:
        row = []
        for tag in ("main", "sub"):
            seg, start, end = windows[tag]
            blended = blend(seg, w)
            ric = daily_rankic(blended, "score")
            row.append((ric["mean_rankic"], ric["rankicir"]))
        print(f"{w:<5.2f}  {row[0][0]:+.4f}   {row[0][1]:.3f}      {row[1][0]:+.4f}   {row[1][1]:.3f}")
        if best is None or row[0][0] > best[1]:
            best = (w, row[0][0])

    w = best[0]
    print(f"\nselected w={w} on main window")
    for tag in args.ir.split(",") if args.ir else []:
        seg, start, end = windows[tag]
        blended = blend(seg, w)
        ir = ir_backtest(blended, "score", str(pd.Timestamp(str(start)).date()),
                         str(pd.Timestamp(str(end)).date()))
        print(f"[{tag}] IR={ir.get('information_ratio')} TE={ir.get('tracking_error')} "
              f"超额={ir.get('annualized_excess_return')}")


if __name__ == "__main__":
    main()
