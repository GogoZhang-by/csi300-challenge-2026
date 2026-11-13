"""
双窗口 OOS 验证脚手架（固化门槛）：
- 主窗 2022 全年（train ≤ 2021-12-31）
- 副窗 2021H2（train ≤ 2021-06-30）
- RankIC = 每日截面 Spearman(score, target)  [target = 次日 O2O，官方 RankIC 口径]
- IR    = 官方 BacktestEngine (top_k=50, rebalance=5, holding=1)

用法：
  from dual_window_validate import evaluate_config
  evaluate_config("基线", cols, "target_rank")
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import get_feature_columns
from train import lgb_params
from skd_backtest import BacktestEngine, OptimizerConfig

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data_temp" / "Data")
CACHE = str(ROOT / "result" / "feature_cache.parquet")
SUB_TEST = str(ROOT / "submission_test")

_df = pd.read_parquet(CACHE)
_df["target_rank"] = _df.groupby("日期")["target"].rank(pct=True) - 0.5
_BASE_COLS = get_feature_columns()

_MASK = (
    _df["target"].notna() & _df["target_rank"].notna()
    & _df["is_suspend"].eq(0) & _df["close"].gt(0)
    & _df["target"].abs().le(0.25) & _df["list_days"].ge(60)
)
_clean = _df.loc[_MASK].copy()


def daily_rankic(seg: pd.DataFrame, score_col: str) -> dict:
    ics = []
    for d, g in seg.groupby("日期"):
        if len(g) >= 50:
            v = spearmanr(g[score_col], g["target"])[0]
            if np.isfinite(v):
                ics.append(v)
    ics = np.array(ics)
    mean = float(ics.mean())
    std = float(ics.std())
    return {
        "mean_rankic": mean,
        "std_rankic": std,
        "rankicir": mean / std if std > 0 else np.nan,
        "positive_ratio": float((ics > 0).mean()),
        "n_days": int(len(ics)),
    }


def train_model(train_end: int, params: dict, label_col: str = "target_rank",
                feature_cols: list | None = None, nround: int = 175):
    fc = feature_cols if feature_cols is not None else _BASE_COLS
    tr = _clean.loc[(_clean["日期"] <= train_end) & _clean[label_col].notna()]
    return lgb.train(
        params,
        lgb.Dataset(tr[fc], label=tr[label_col], feature_name=fc),
        num_boost_round=nround,
    )


def ir_backtest(seg: pd.DataFrame, score_col: str, start: str, end: str) -> dict:
    """官方 BacktestEngine (top_k=50)。seg 需含 日期/代码/score。"""
    pred = seg[["日期", "代码", score_col]].rename(
        columns={"日期": "d", "代码": "c", score_col: "score"}
    )
    Path(SUB_TEST, "model").mkdir(parents=True, exist_ok=True)
    pred.to_parquet(Path(SUB_TEST, "model", "_pred.parquet"), index=False)
    Path(SUB_TEST, "model", "metadata.json").write_text(
        json.dumps({"feature_columns": _BASE_COLS, "score_avg_window": 1}), encoding="utf-8"
    )
    eng = BacktestEngine(
        data_dir=DATA, start_date=start, end_date=end,
        submission_dir=SUB_TEST, lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        friendly_output=False,
    )
    r = eng.run()
    return {
        "information_ratio": r.get("information_ratio"),
        "tracking_error": r.get("tracking_error"),
        "annualized_excess_return": r.get("annualized_excess_return"),
        "engine_rankic": r.get("mean_rankic"),
    }


def _postprocess(seg: pd.DataFrame, score_col: str, fn) -> str:
    """对某窗口的预测应用后处理，返回新 score 列名（返回 None 表示跳过）。"""
    if fn is None:
        return score_col
    seg = fn(seg, score_col)
    return score_col


def evaluate_config(
    name: str,
    feature_cols: list | None = None,
    label_col: str = "target_rank",
    params_overrides: dict | None = None,
    nround: int = 175,
    postprocess=None,
    run_ir: bool = True,
):
    """跑双窗口 OOS 并打印结果。返回 (主窗RankIC, 主窗IR, 副窗RankIC, 副窗IR)。"""
    params = lgb_params(42)
    if params_overrides:
        params.update(params_overrides)

    print(f"\n{'='*70}\n实验: {name}\n{'='*70}")
    results = {}

    # ── 主窗 2022 ──────────────────────────────────────────────
    m22 = train_model(20211231, params, label_col, feature_cols, nround)
    seg22 = _df.loc[_df["日期"].between(20220101, 20221231)].copy()
    seg22["score"] = m22.predict(seg22[feature_cols if feature_cols is not None else _BASE_COLS].fillna(0.0))
    sc22 = _postprocess(seg22, "score", postprocess)
    ric22 = daily_rankic(seg22, sc22)
    ir22 = ir_backtest(seg22, sc22, "2022-01-01", "2022-12-31") if run_ir else {}
    results["main_rankic"] = ric22["mean_rankic"]
    results["main_ir"] = ir22.get("information_ratio")
    print(f"[主窗 2022] RankIC={ric22['mean_rankic']:.4f} ICIR={ric22['rankicir']:.3f} "
          f"正IC%={ric22['positive_ratio']:.1%} 天数={ric22['n_days']}")
    if run_ir:
        print(f"             IR={ir22['information_ratio']:.3f} TE={ir22['tracking_error']:.3f} "
              f"超额={ir22['annualized_excess_return']:+.3f}")

    # ── 副窗 2021H2 ────────────────────────────────────────────
    m21 = train_model(20210630, params, label_col, feature_cols, nround)
    seg21 = _df.loc[_df["日期"].between(20210701, 20211231)].copy()
    seg21["score"] = m21.predict(seg21[feature_cols if feature_cols is not None else _BASE_COLS].fillna(0.0))
    sc21 = _postprocess(seg21, "score", postprocess)
    ric21 = daily_rankic(seg21, sc21)
    ir21 = ir_backtest(seg21, sc21, "2021-07-01", "2021-12-31") if run_ir else {}
    results["sub_rankic"] = ric21["mean_rankic"]
    results["sub_ir"] = ir21.get("information_ratio")
    print(f"[副窗 2021H2] RankIC={ric21['mean_rankic']:.4f} ICIR={ric21['rankicir']:.3f} "
          f"正IC%={ric21['positive_ratio']:.1%} 天数={ric21['n_days']}")
    if run_ir:
        print(f"               IR={ir21['information_ratio']:.3f} TE={ir21['tracking_error']:.3f} "
              f"超额={ir21['annualized_excess_return']:+.3f}")

    return results


if __name__ == "__main__":
    evaluate_config("基线 rank1 回归")
