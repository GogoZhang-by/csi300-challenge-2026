"""Evaluate a retrained FNR model on a date window with honest OOS metrics.

Calls the vendored champion ``InferenceModel`` once per trading day (it only
consumes the trailing 120 exchange dates), then reports daily RankIC and the
official ``skd_backtest`` Information Ratio.

Usage:
  python scripts/fnr_eval.py --model-dir result/fnr_cut2021_12 --start 20220101 --end 20221231 --tag main
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from champion.inference import InferenceModel  # noqa: E402
from champion.fnr.inference.rank_merge import merge_scores  # noqa: E402
from dual_window_validate import _df, daily_rankic, ir_backtest  # noqa: E402

DATA = ROOT / "data_temp" / "Data"
TABLES = ("MarketData", "Factor33_winsor", "Barra_factor")
LOOKBACK = 130


def load_tables(start: int, end: int) -> dict[str, pd.DataFrame]:
    # Include ~8 months of history so the model's 120-day window is available
    # on the first evaluation date.
    sy, sm = start // 10000, (start // 100) % 100
    sm -= 8
    while sm <= 0:
        sm += 12
        sy -= 1
    first = sy * 100 + sm
    out = {}
    for name in TABLES:
        parts = []
        for p in sorted((DATA / name).glob("*/*/*.parquet")):
            if first <= int(p.stem) <= end // 100:
                parts.append(pd.read_parquet(p))
        out[name] = pd.concat(parts, ignore_index=True).sort_values(["日期", "代码"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--limit", type=int, default=0, help="only first N dates (debug)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-ir", action="store_true")
    ap.add_argument("--dump-internals", action="store_true",
                    help="also save rh50 / K3 sub-scores for head_count tuning")
    args = ap.parse_args()

    tables = load_tables(args.start, args.end)
    all_dates = np.sort(tables["MarketData"]["日期"].unique())
    dates = all_dates[(all_dates >= args.start) & (all_dates <= args.end)]
    if args.limit:
        dates = dates[: args.limit]
    print(f"dates={len(dates)} {dates[0]}..{dates[-1]}", flush=True)

    model = InferenceModel(str(args.model_dir))
    rows = []
    internals = []
    for i, d in enumerate(dates):
        pos = int(np.searchsorted(all_dates, d))
        window = set(all_dates[max(0, pos - LOOKBACK + 1): pos + 1])
        data = {k: v.loc[v["日期"].isin(window)].copy() for k, v in tables.items()}
        stamp = pd.Timestamp(str(d)).strftime("%Y-%m-%d")
        if args.dump_internals:
            raw = model._predict(stamp, data)
            internals.append(pd.DataFrame({
                "日期": d, "代码": raw["code"].to_numpy(),
                "rh50": raw["rh50_score"].to_numpy(), "k3": raw["score"].to_numpy(),
            }))
            pred = merge_scores(
                raw[["date", "code", "rh50_score"]].rename(columns={"rh50_score": "score"}),
                raw[["date", "code", "score"]],
                model.head_count,
            )[["date", "code", "score"]].rename(columns={"code": "代码"})
        else:
            pred = model.predict(stamp, data).rename(columns={"date": "日期", "code": "代码"})
        pred["日期"] = d
        rows.append(pred[["日期", "代码", "score"]])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(dates)} {d}", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    if internals:
        internal_path = ROOT / "result" / f"fnr_internal_{args.tag}.parquet"
        pd.concat(internals, ignore_index=True).to_parquet(internal_path, index=False)
        print(f"internals -> {internal_path}", flush=True)

    out = args.out or (ROOT / "result" / f"fnr_scores_{args.tag}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out, index=False)
    print(f"predictions -> {out} rows={len(scores)}", flush=True)

    seg = _df.loc[_df["日期"].between(args.start, args.end)].merge(
        scores, on=["日期", "代码"], how="left", validate="one_to_one"
    )
    ric = daily_rankic(seg, "score")
    print(f"[{args.tag}] RankIC={ric['mean_rankic']:.4f} ICIR={ric['rankicir']:.3f} "
          f"正IC%={ric['positive_ratio']:.1%} 天数={ric['n_days']}")
    if args.skip_ir:
        return
    ir = ir_backtest(seg, "score", str(pd.Timestamp(str(args.start)).date()),
                     str(pd.Timestamp(str(args.end)).date()))
    print(f"[{args.tag}] IR={ir.get('information_ratio')} TE={ir.get('tracking_error')} "
          f"超额={ir.get('annualized_excess_return')}")


if __name__ == "__main__":
    main()
