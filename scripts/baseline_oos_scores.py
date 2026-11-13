"""Dump honest OOS scores of the 71-feature LightGBM baseline for blending.

Trains a fresh model at each cutoff (no early stopping, no peeking) and writes
per-day scores for the corresponding evaluation window.

Usage:
  python scripts/baseline_oos_scores.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dual_window_validate import _BASE_COLS, _df, train_model  # noqa: E402
from train import lgb_params  # noqa: E402

WINDOWS = [
    ("main", 20211231, 20220101, 20221231),
    ("sub", 20210630, 20210701, 20211231),
]


def main():
    out_dir = ROOT / "result"
    out_dir.mkdir(exist_ok=True)
    for tag, cutoff, start, end in WINDOWS:
        model = train_model(cutoff, lgb_params(42), "target_rank", _BASE_COLS, 175)
        seg = _df.loc[_df["日期"].between(start, end)].copy()
        seg["score71"] = model.predict(seg[_BASE_COLS].fillna(0.0))
        path = out_dir / f"baseline_scores_{tag}.parquet"
        seg[["日期", "代码", "score71"]].to_parquet(path, index=False)
        print(f"{tag}: {len(seg)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
