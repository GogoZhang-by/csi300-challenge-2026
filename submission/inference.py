"""
推理入口 — 完全自包含（无外部依赖），内联特征构建代码。
提交目录结构:
  submission/
    inference.py   <-- 本文件
    train.py
    requirements.txt
    model/
      lgb_reg.txt
      metadata.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# ── 内联特征列清单（与训练产出 metadata.json 保持一致）─────────────────────
_FACTOR_FEATURES = (
    "turn", "change_ratio", "daily_return", "momentum_5", "volatility_5",
    "rsi_12", "kdj_k_9_3_3", "kdj_d_9_3_3", "bias_20", "cci_14",
    "total_market_cap", "float_market_cap", "pe_ttm", "pb", "ps_ttm",
    "roe_avg_ttm", "roa_avg_ttm", "gross_profit_rate_ttm",
    "net_profit_rate_ttm", "debt_to_asset_lf", "current_ratio_lf",
    "beta_000300SH_22", "list_days", "is_suspend", "is_st", "is_new",
)
_EXTRA_FACTOR_FEATURES = (
    "reversal_5", "sma_20", "ema_20", "macd_diff_12_26_9",
    "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14",
)
_BARRA_FEATURES = (
    "市值", "贝塔", "动量", "残差波动", "非线性市值",
    "账面市值比", "流动性", "盈利", "成长", "杠杆",
)
_TS_FEATURES = (
    "ret_1", "mom_5", "mom_20", "mom_60",
    "vol_20", "vol_60", "max_ret_20",
    "overnight_vol_20", "hl_range_20", "amt_zscore_120", "intraday_ret",
)
# 修改点：新增截面均值特征（Kaggle冠军trick）| 优化目标：提升RankIC | 预期作用：+0.005 IC
_DMEAN_FEATURES = (
    "mom_5", "mom_20", "mom_60", "reversal_5", "momentum_5",
    "ret_1", "daily_return", "change_ratio", "intraday_ret",
    "vol_20", "barra_动量",
)


def _get_feature_columns() -> list[str]:
    rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                                "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
    dmean = [f"{c}_dmean" for c in _DMEAN_FEATURES]
    return [
        *_FACTOR_FEATURES, *_EXTRA_FACTOR_FEATURES,
        *(f"barra_{n}" for n in _BARRA_FEATURES),
        *_TS_FEATURES,
        *rel,
        *dmean,
    ]


# ── 内联特征构建（与训练完全一致）───────────────────────────────────────────
def _build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market = data["MarketData"].copy()
    factors = data["Factor33_winsor"].copy()
    barra = data["Barra_factor"].copy()

    market = market.sort_values(["代码", "日期"]).reset_index(drop=True)
    g = market.groupby("代码", sort=False)
    prev_close = g["close"].shift(1)

    market["ret_1"] = market["close"] / prev_close - 1.0
    market["mom_5"] = market["close"] / g["close"].shift(5) - 1.0
    market["mom_20"] = market["close"] / g["close"].shift(20) - 1.0
    market["mom_60"] = market["close"] / g["close"].shift(60) - 1.0
    market["vol_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=15).std())
    market["vol_60"] = g["ret_1"].transform(lambda s: s.rolling(60, min_periods=45).std())
    market["max_ret_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=15).max())
    market["overnight_ret"] = market["open"] / prev_close - 1.0
    market["overnight_vol_20"] = g["overnight_ret"].transform(lambda s: s.rolling(20, min_periods=15).std())
    market["intraday_ret"] = market["close"] / market["open"] - 1.0
    hl = (market["high"] - market["low"]) / prev_close
    market["hl_range_20"] = hl.groupby(market["代码"], sort=False).transform(lambda s: s.rolling(20, min_periods=15).mean())
    amt_mean = g["amount"].transform(lambda s: s.rolling(120, min_periods=90).mean())
    amt_std = g["amount"].transform(lambda s: s.rolling(120, min_periods=90).std())
    market["amt_zscore_120"] = (market["amount"] - amt_mean) / amt_std.replace(0, np.nan)

    ts_cols = ["日期", "代码", *_TS_FEATURES]
    base = barra.loc[:, ["日期", "代码", *_BARRA_FEATURES]].merge(
        factors.loc[:, ["日期", "代码", *(_FACTOR_FEATURES + _EXTRA_FACTOR_FEATURES)]],
        on=["日期", "代码"], how="left", validate="one_to_one", sort=False,
    ).merge(
        market.loc[:, ts_cols],
        on=["日期", "代码"], how="left", validate="one_to_one", sort=False,
    )

    base = base.merge(market.loc[:, ["日期", "代码", "close"]], on=["日期", "代码"],
                      how="left", validate="one_to_one", sort=False)
    for name in ("sma_20", "ema_20", "macd_diff_12_26_9", "macd_dea_12_26_9",
                 "macd_hist_12_26_9", "atr_14"):
        base[f"{name}_rel"] = pd.to_numeric(base[name], errors="coerce") / base["close"]

    base = base.rename(columns={n: f"barra_{n}" for n in _BARRA_FEATURES})

    feat_cols = [c for c in base.columns if c not in ("日期", "代码", "名称", "close")]
    for c in feat_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base[feat_cols] = base[feat_cols].replace([np.inf, -np.inf], np.nan)

    # 截面均值特征（冠军 trick）：对核心特征按日做截面均值
    for col in _DMEAN_FEATURES:
        if col in base.columns:
            base[f"{col}_dmean"] = base.groupby("日期")[col].transform("mean")

    return base


# ── InferenceModel 类 ──────────────────────────────────────────────────────
class InferenceModel:
    def __init__(self, model_dir: str):
        directory = Path(model_dir)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.cols = meta["feature_columns"]
        self.avg_window = int(meta.get("score_avg_window", 20))
        self.model = lgb.Booster(model_file=str(directory / "lgb_reg.txt"))

    def predict(self, as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        feats = _build_features(data)

        for c in feats.columns:
            if pd.api.types.is_numeric_dtype(feats[c]):
                feats[c] = feats[c].replace([np.inf, -np.inf], np.nan)
                feats[c] = feats[c].fillna(0.0)

        day = int(as_of_date.replace("-", ""))

        # 对窗口内所有日期打分，再按股票对最近 avg_window 个交易日的
        # 预测值取滚动平均（低换手平滑；仅用 ≤ 当日数据，无跨调用状态）
        matrix = feats.reindex(columns=self.cols).fillna(0.0)
        feats["_raw"] = self.model.predict(matrix)
        feats = feats.sort_values(["代码", "日期"])
        feats["_score"] = feats.groupby("代码", sort=False)["_raw"].transform(
            lambda s: s.rolling(self.avg_window, min_periods=1).mean()
        )

        row = feats.loc[feats["日期"].eq(day)].copy()
        scores = row["_score"].to_numpy()

        if not np.isfinite(scores).all():
            raise ValueError("non-finite scores produced")

        return pd.DataFrame({
            "date": as_of_date,
            "code": row["代码"].to_numpy(),
            "score": scores,
        })
