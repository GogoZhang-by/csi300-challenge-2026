"""
特征工程 — 经消融实验(diagnose5)验证的固化特征集：
- baseline 49 个原始因子（Factor33 + Barra + _rel）
- 12 个时序扩展特征（来自 MarketData 历史 ≤120 日窗口）
- 已剔除损害 OOS 的 rank/susp_limit 特征

所有特征均可用 ≤120 交易日历史计算，满足赛题 look-back 约束。
训练与推理共用同一函数，保证口径一致。
"""
import numpy as np
import pandas as pd


# ── baseline 已有特征（Factor33 + Barra）───────────────────────────────────
FACTOR_FEATURES = (
    "turn", "change_ratio", "daily_return", "momentum_5", "volatility_5",
    "rsi_12", "kdj_k_9_3_3", "kdj_d_9_3_3", "bias_20", "cci_14",
    "total_market_cap", "float_market_cap", "pe_ttm", "pb", "ps_ttm",
    "roe_avg_ttm", "roa_avg_ttm", "gross_profit_rate_ttm",
    "net_profit_rate_ttm", "debt_to_asset_lf", "current_ratio_lf",
    "beta_000300SH_22", "list_days", "is_suspend", "is_st", "is_new",
)
EXTRA_FACTOR_FEATURES = (
    "reversal_5", "sma_20", "ema_20", "macd_diff_12_26_9",
    "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14",
)
BARRA_FEATURES = (
    "市值", "贝塔", "动量", "残差波动", "非线性市值",
    "账面市值比", "流动性", "盈利", "成长", "杠杆",
)

# 经消融验证的时序扩展特征（diagnose5: +时序组 1dIC 0.0290 > baseline49 0.0279）
TS_FEATURES = (
    "ret_1", "mom_5", "mom_20", "mom_60",
    "vol_20", "vol_60", "max_ret_20",
    "overnight_vol_20", "hl_range_20", "amt_zscore_120", "intraday_ret",
)

# 修改点：新增截面均值特征（Kaggle冠军trick）| 优化目标：提升RankIC | 预期作用：+0.005 IC
# 对核心量价特征计算每日横截面均值，让模型感知市场/风格整体水平
DMEAN_FEATURES = (
    "mom_5", "mom_20", "mom_60", "reversal_5", "momentum_5",
    "ret_1", "daily_return", "change_ratio", "intraday_ret",
    "vol_20", "barra_动量",
)


def build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    输入: {"MarketData": df, "Factor33_winsor": df, "Barra_factor": df}
    输出: 全部日期的特征表 [日期, 代码, 特征...]
    训练时传入全量历史；推理时传入平台给的 ≤120 日窗口。
    """
    market = data["MarketData"].copy()
    factors = data["Factor33_winsor"].copy()
    barra = data["Barra_factor"].copy()

    market = market.sort_values(["代码", "日期"]).reset_index(drop=True)

    # ── 时序特征（基于后复权价格，满足 ≤120 日窗口）─────────────────────────
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
    market["overnight_vol_20"] = g["overnight_ret"].transform(
        lambda s: s.rolling(20, min_periods=15).std()
    )
    market["intraday_ret"] = market["close"] / market["open"] - 1.0
    hl = (market["high"] - market["low"]) / prev_close
    market["hl_range_20"] = hl.groupby(market["代码"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    amt_mean = g["amount"].transform(lambda s: s.rolling(120, min_periods=90).mean())
    amt_std = g["amount"].transform(lambda s: s.rolling(120, min_periods=90).std())
    market["amt_zscore_120"] = (market["amount"] - amt_mean) / amt_std.replace(0, np.nan)

    # ── 合并基础因子（以 Barra 当日成分股为准）─────────────────────────────
    ts_cols = ["日期", "代码", *TS_FEATURES]
    base = barra.loc[:, ["日期", "代码", *BARRA_FEATURES]].merge(
        factors.loc[:, ["日期", "代码", *(FACTOR_FEATURES + EXTRA_FACTOR_FEATURES)]],
        on=["日期", "代码"], how="left", validate="one_to_one", sort=False,
    ).merge(
        market.loc[:, ts_cols],
        on=["日期", "代码"], how="left", validate="one_to_one", sort=False,
    )

    # 价格尺度因子 / close（与 baseline 相同的 _rel 处理）
    base = base.merge(
        market.loc[:, ["日期", "代码", "close"]], on=["日期", "代码"],
        how="left", validate="one_to_one", sort=False,
    )
    for name in ("sma_20", "ema_20", "macd_diff_12_26_9", "macd_dea_12_26_9",
                 "macd_hist_12_26_9", "atr_14"):
        base[f"{name}_rel"] = pd.to_numeric(base[name], errors="coerce") / base["close"]

    # Barra 列名加前缀
    base = base.rename(columns={n: f"barra_{n}" for n in BARRA_FEATURES})

    # 数值化 + inf 清理
    feat_cols = [c for c in base.columns if c not in ("日期", "代码", "名称", "close")]
    for c in feat_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base[feat_cols] = base[feat_cols].replace([np.inf, -np.inf], np.nan)

    # 截面均值特征（冠军 trick）：对核心特征按日做截面均值，让模型感知市场整体水平
    for col in DMEAN_FEATURES:
        if col in base.columns:
            base[f"{col}_dmean"] = base.groupby("日期")[col].transform("mean")

    return base


def get_feature_columns() -> list[str]:
    rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                                "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
    dmean = [f"{c}_dmean" for c in DMEAN_FEATURES]
    return [
        *FACTOR_FEATURES, *EXTRA_FACTOR_FEATURES,
        *(f"barra_{n}" for n in BARRA_FEATURES),
        *TS_FEATURES,
        *rel,
        *dmean,
    ]
