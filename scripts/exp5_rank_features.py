"""建议5 实验：截面 rank(pct) 特征预处理（双窗口 OOS）"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import lightgbm as lgb

from dual_window_validate import (
    _df, _clean, _BASE_COLS, daily_rankic, ir_backtest, lgb_params,
)


def _rank_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _BASE_COLS:
        out[c] = out.groupby("日期")[c].rank(pct=True)
    return out


def _train(train_end: int):
    tr = _clean.loc[_clean["日期"].le(train_end)]
    trf = _rank_feats(tr[["日期", *_BASE_COLS]])
    ds = lgb.Dataset(trf[_BASE_COLS], label=tr["target_rank"].to_numpy(), feature_name=_BASE_COLS)
    return lgb.train(lgb_params(42), ds, num_boost_round=175)


def _run(train_end: int, test_start: int, test_end: int, tag: str):
    m = _train(train_end)
    seg = _df.loc[_df["日期"].between(test_start, test_end)].copy()
    segf = _rank_feats(seg[["日期", *_BASE_COLS]])[_BASE_COLS].fillna(0.5)
    seg["score"] = m.predict(segf)
    ric = daily_rankic(seg, "score")
    ir = ir_backtest(seg, "score", f"{test_start//10000}-{test_start%10000//100:02d}-{test_start%100:02d}",
                     f"{test_end//10000}-{test_end%10000//100:02d}-{test_end%100:02d}")
    print(f"[{tag}] RankIC={ric['mean_rankic']:.4f} ICIR={ric['rankicir']:.3f} "
          f"正IC%={ric['positive_ratio']:.1%} | IR={ir['information_ratio']:.3f} TE={ir['tracking_error']:.3f}")
    return ric["mean_rankic"], ir["information_ratio"]


if __name__ == "__main__":
    print("=" * 70)
    print("建议5：截面 rank(pct) 特征预处理（71维特征逐日转 rank(pct)）")
    print("=" * 70)
    _run(20211231, 20220101, 20221231, "主窗 2022")
    _run(20210630, 20210701, 20211231, "副窗 2021H2")
