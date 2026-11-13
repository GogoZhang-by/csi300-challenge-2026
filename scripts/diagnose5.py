"""诊断5：特征消融 — 按组添加扩展特征，看每组对2022 OOS的影响"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from validate import RankICEvaluator
from features import (get_feature_columns, FACTOR_FEATURES, EXTRA_FACTOR_FEATURES,
                      BARRA_FEATURES, TS_FEATURES, SUSP_LIMIT_FEATURES, RANK_FEATURES)

df = pd.read_parquet("result/feature_cache.parquet")
rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                            "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
BASE = list(FACTOR_FEATURES) + [f"barra_{n}" for n in BARRA_FEATURES] + list(EXTRA_FACTOR_FEATURES) + rel

mask = (df["target"].notna() & df["target_5"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
test = clean.loc[clean["日期"].gt(20211231)]

params = {"objective": "regression", "metric": "l2", "verbosity": -1,
          "learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
          "min_data_in_leaf": 100, "feature_fraction": 0.85,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "seed": 42, "num_threads": 8, "deterministic": True, "force_col_wise": True}

def run(cols, tag):
    m = lgb.train(params, lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(test[cols], label=test["target_5"], feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    s = m.predict(test[cols], num_iteration=m.best_iteration)
    t = test.assign(s=s)
    r1 = RankICEvaluator.evaluate(t.rename(columns={"日期": "date"}), label_col="target", score_col="s")
    r5 = RankICEvaluator.evaluate(t.rename(columns={"日期": "date"}), label_col="target_5", score_col="s")
    print(f"{tag:30s} n={len(cols):3d} iter={m.best_iteration:3d} | 1dIC={r1['mean_rankic']:.4f} IR={r1['rankicir']:.3f} | 5dIC={r5['mean_rankic']:.4f} IR={r5['rankicir']:.3f}")
    return m

run(BASE, "baseline49")
run(BASE + list(SUSP_LIMIT_FEATURES), "+停牌涨停")
run(BASE + list(TS_FEATURES), "+时序")
run(BASE + [f"{n}_rank" for n in RANK_FEATURES], "+截面rank")
run(BASE + list(SUSP_LIMIT_FEATURES) + [f"{n}_rank" for n in RANK_FEATURES], "+停牌涨停+rank")
run(get_feature_columns(), "全部83")

# 单特征 IC 扫描（train期 vs test期符号一致性）
print("\n=== 扩展特征逐因子 IC（train期 / test期, 对target_5）===")
ext = list(TS_FEATURES) + list(SUSP_LIMIT_FEATURES) + [f"{n}_rank" for n in RANK_FEATURES]
from scipy.stats import spearmanr
rows = []
for c in ext:
    ic_tr = train.groupby("日期").apply(lambda g: spearmanr(g[c], g["target_5"])[0] if g[c].notna().sum()>30 else np.nan).mean()
    ic_te = test.groupby("日期").apply(lambda g: spearmanr(g[c], g["target_5"])[0] if g[c].notna().sum()>30 else np.nan).mean()
    rows.append((c, ic_tr, ic_te, np.sign(ic_tr)==np.sign(ic_te)))
res = pd.DataFrame(rows, columns=["feature", "ic_train", "ic_test", "sign_match"]).sort_values("ic_test")
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
