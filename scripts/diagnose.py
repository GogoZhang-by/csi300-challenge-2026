"""诊断2：截面rank标签 + 全量训练池 vs Barra池"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from validate import RankICEvaluator
from data_loader import DataLoader
from features import build_features, FACTOR_FEATURES, EXTRA_FACTOR_FEATURES, BARRA_FEATURES

# 用全量 Factor33 并集重建特征（不局限 Barra 300）
cache = "result/feature_cache_full.parquet"
import os
if os.path.exists(cache):
    full = pd.read_parquet(cache)
else:
    dl = DataLoader("data_temp/Data")
    md = dl.load_market_data()
    f33 = dl.load_factor33()
    bar = dl.load_barra()
    # 改造：build_features 以 f33 并集为股票池 —— 手动构建
    data = {"MarketData": md, "Factor33_winsor": f33, "Barra_factor": bar}
    feats_barra = build_features(data)  # Barra 池版本（含扩展特征）
    # 全量池版本：把 f33 作为"barra"左表
    data2 = {"MarketData": md, "Factor33_winsor": f33,
             "Barra_factor": f33[["日期", "代码"]].assign(**{n: np.nan for n in BARRA_FEATURES})}
    # 上面会把 barra 特征置 NaN，不理想。改为：f33 并集 + 真 barra left join
    feats_full_base = f33.merge(bar[["日期", "代码", *BARRA_FEATURES]], on=["日期", "代码"], how="left")
    # 扩展时序特征从 feats_barra 中按(日期,代码)补齐 —— 但 feats_barra 只有300只。
    # 简化：全量池只用 Factor33 原始49特征，不加扩展时序特征
    labels = dl.build_labels(md)
    full = feats_full_base.merge(labels, on=["日期", "代码"], how="left")
    full.to_parquet(cache)

rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                            "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
# 补 _rel 特征
md_close = DataLoader("data_temp/Data").load_market_data()[["日期", "代码", "close"]]
full = full.merge(md_close, on=["日期", "代码"], how="left")
for n in ("sma_20", "ema_20", "macd_diff_12_26_9", "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14"):
    full[f"{n}_rel"] = full[n] / full["close"]
full = full.rename(columns={n: f"barra_{n}" for n in BARRA_FEATURES})
BASELINE_COLS = list(FACTOR_FEATURES) + [f"barra_{n}" for n in BARRA_FEATURES] + list(EXTRA_FACTOR_FEATURES) + rel

# baseline 式清洗：仅剔除 停牌/缺价/非正价格
mask = (full["target"].notna() & full["target_5"].notna()
        & full["is_suspend"].eq(0) & full["close"].gt(0))
clean = full.loc[mask].copy()
# 截面 rank 标签
clean["rank_label_5"] = clean.groupby("日期")["target_5"].rank(pct=True)

train = clean.loc[clean["日期"].le(20201231)]
val = clean.loc[clean["日期"].gt(20201231) & clean["日期"].le(20211231)]
test = clean.loc[clean["日期"].gt(20211231)]

params = {"objective": "regression", "metric": "l2", "verbosity": -1,
          "learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
          "min_data_in_leaf": 100, "feature_fraction": 0.85,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "seed": 42, "num_threads": 8, "deterministic": True, "force_col_wise": True}

def ev(df_, col, label="target"):
    return RankICEvaluator.evaluate(df_.rename(columns={"日期": "date"}), label_col=label, score_col=col)["mean_rankic"]

for lab in ["rank_label_5", "target_5"]:
    m = lgb.train(params, lgb.Dataset(train[BASELINE_COLS], label=train[lab], feature_name=BASELINE_COLS),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(val[BASELINE_COLS], label=val[lab], feature_name=BASELINE_COLS)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    val = val.assign(s=m.predict(val[BASELINE_COLS], num_iteration=m.best_iteration))
    test = test.assign(s=m.predict(test[BASELINE_COLS], num_iteration=m.best_iteration))
    # 评估只在 Barra 成分股上（官方口径）
    bar_universe = set(pd.read_parquet("result/feature_cache.parquet")["代码"].unique())
    val_b = val[val["代码"].isin(bar_universe)]
    test_b = test[test["代码"].isin(bar_universe)]
    print(f"train_label={lab}: best_iter={m.best_iteration}")
    print(f"  val(Barra池):  1dIC={ev(val_b,'s'):.4f}  5dIC={ev(val_b,'s','target_5'):.4f}")
    print(f"  test(Barra池): 1dIC={ev(test_b,'s'):.4f}  5dIC={ev(test_b,'s','target_5'):.4f}")
