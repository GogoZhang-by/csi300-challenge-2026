"""
训练脚本 — 完全自包含（无外部依赖），可从公开 2016-2022 数据复现 model/。
用法: python train.py --data-dir <数据根目录>
数据根目录下应包含 MarketData/ Factor33_winsor/ Barra_factor/ 三个子目录。
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# ── 特征列清单（与 inference.py 保持一致）───────────────────────────────────
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
# 截面均值特征（冠军 trick）
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


# ── 数据加载 ────────────────────────────────────────────────────────────────
def _load_table(root: Path, table: str) -> pd.DataFrame:
    paths = sorted(root.joinpath(table).rglob("*.parquet"))
    frames = [pd.read_parquet(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["日期", "代码"]).reset_index(drop=True)


def _build_labels(market: pd.DataFrame) -> pd.DataFrame:
    df = market[["日期", "代码", "open"]].copy()
    df = df.sort_values(["代码", "日期"]).reset_index(drop=True)
    g = df.groupby("代码")["open"]
    df["o1"] = g.shift(-1)
    df["o2"] = g.shift(-2)
    df["o6"] = g.shift(-6)
    df["o21"] = g.shift(-21)
    df["target"] = df["o2"] / df["o1"] - 1.0
    df["target_5"] = df["o6"] / df["o1"] - 1.0
    df["target_20"] = df["o21"] / df["o1"] - 1.0
    return df[["日期", "代码", "target", "target_5", "target_20"]]


# ── 特征构建（与 inference.py 完全一致）─────────────────────────────────────
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

    # 截面均值特征（冠军 trick）
    for col in _DMEAN_FEATURES:
        if col in base.columns:
            base[f"{col}_dmean"] = base.groupby("日期")[col].transform("mean")

    return base


def _lgb_params(seed: int) -> dict:
    return {
        "objective": "regression", "metric": "l2", "verbosity": -1,
        "learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
        "min_data_in_leaf": 100, "feature_fraction": 0.85,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "seed": seed, "num_threads": 8, "deterministic": True,
        "force_col_wise": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("model"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = _get_feature_columns()
    print(f"特征数: {len(cols)}")

    md = _load_table(args.data_dir, "MarketData")
    f33 = _load_table(args.data_dir, "Factor33_winsor")
    bar = _load_table(args.data_dir, "Barra_factor")

    data = {"MarketData": md, "Factor33_winsor": f33, "Barra_factor": bar}
    feats = _build_features(data)
    labels = _build_labels(md)
    df = feats.merge(labels, on=["日期", "代码"], how="left", validate="one_to_one")

    # 修改点：标签口径对齐评测——评测RankIC是次日(H=1)O2O收益排序，标签也用1日收益的截面rank
    # 优化目标：RankIC（70%权重主指标） | 预期作用：消除20日标签与1日评测的错位（公榜RankIC 0.0141的根因）
    df["target_rank"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

    mask = (
        df["target"].notna() & df["target_rank"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25)
        & df["list_days"].ge(60)
    )
    clean = df.loc[mask].copy()
    print(f"样本: 总{len(df)} -> 清洗后{len(clean)}")

    train = clean.loc[clean["日期"].le(20211231)]
    val = clean.loc[clean["日期"].between(20220101, 20221231)]

    m = lgb.train(
        _lgb_params(args.seed),
        lgb.Dataset(train[cols], label=train["target_rank"], feature_name=cols),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val[cols], label=val["target_rank"], feature_name=cols)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = m.best_iteration
    print(f"best_iter={best_iter}")

    final = clean.loc[clean["日期"].le(20221231)]
    m_final = lgb.train(
        _lgb_params(args.seed),
        lgb.Dataset(final[cols], label=final["target_rank"], feature_name=cols),
        num_boost_round=best_iter,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    m_final.save_model(str(args.out / "lgb_reg.txt"))
    meta = {
        "feature_columns": cols,
        "label_col": "target_rank",
        "seed": args.seed,
        "best_iter": best_iter,
        "score_avg_window": 1,
        "train_range": [20160104, 20221231],
    }
    (args.out / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 模型已保存到 {args.out}")


if __name__ == "__main__":
    main()
