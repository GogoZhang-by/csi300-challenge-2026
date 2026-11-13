"""建议2 实验：回归 → lambdarank 成对排序损失（双窗口 OOS）"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import lightgbm as lgb

from dual_window_validate import (
    _df, _clean, _BASE_COLS, daily_rankic, ir_backtest, lgb_params,
)

N_BUCKET = 300  # 直接按日截面排名作为整数相关性等级，最细粒度成对比较


def _lambdarank_params():
    p = lgb_params(42)
    p.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [300],
        "label_gain": ",".join(str(i) for i in range(N_BUCKET)),
        "lambdarank_truncation_level": 300,
    })
    return p


def _build_train(train_end: int):
    tr = _clean.loc[_clean["日期"].le(train_end)].copy()
    tr = tr.sort_values("日期", kind="stable")
    # 每日截面排名 → 0..N-1 整数相关性等级（越高越好）
    tr["grade"] = (tr.groupby("日期")["target"].rank(method="average") - 1).astype(int)
    groups = tr.groupby("日期", sort=False).size().to_numpy()
    ds = lgb.Dataset(
        tr[_BASE_COLS], label=tr["grade"], group=groups, feature_name=_BASE_COLS
    )
    return tr, ds


def _run_window(train_end: int, test_start: int, test_end: int, tag: str):
    _, ds = _build_train(train_end)
    m = lgb.train(_lambdarank_params(), ds, num_boost_round=175)
    seg = _df.loc[_df["日期"].between(test_start, test_end)].copy()
    seg["score"] = m.predict(seg[_BASE_COLS].fillna(0.0))
    ric = daily_rankic(seg, "score")
    ir = ir_backtest(seg, "score", f"{test_start//10000}-{test_start%10000//100:02d}-{test_start%100:02d}",
                     f"{test_end//10000}-{test_end%10000//100:02d}-{test_end%100:02d}")
    print(f"[{tag}] RankIC={ric['mean_rankic']:.4f} ICIR={ric['rankicir']:.3f} "
          f"正IC%={ric['positive_ratio']:.1%} | IR={ir['information_ratio']:.3f} TE={ir['tracking_error']:.3f}")
    return ric["mean_rankic"], ir["information_ratio"]


if __name__ == "__main__":
    print("=" * 70)
    print("建议2：lambdarank 成对排序损失 (grade=日截面排名0..N-1, group=按日)")
    print("=" * 70)
    _run_window(20211231, 20220101, 20221231, "主窗 2022")
    _run_window(20210630, 20210701, 20211231, "副窗 2021H2")
