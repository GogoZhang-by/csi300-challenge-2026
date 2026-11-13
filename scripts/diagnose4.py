"""诊断4：同一训练协议下，baseline49特征 vs 全量扩展特征（含停牌/涨停/时序/rank）"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from validate import RankICEvaluator
from features import get_feature_columns, FACTOR_FEATURES, EXTRA_FACTOR_FEATURES, BARRA_FEATURES

df = pd.read_parquet("result/feature_cache.parquet")  # Barra池 + 扩展特征 + 标签

rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                            "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
BASELINE_COLS = list(FACTOR_FEATURES) + [f"barra_{n}" for n in BARRA_FEATURES] + list(EXTRA_FACTOR_FEATURES) + rel
FULL_COLS = get_feature_columns()

# baseline 式清洗（尽量一致）
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

def ev(df_, col, label="target"):
    r = RankICEvaluator.evaluate(df_.rename(columns={"日期": "date"}), label_col=label, score_col=col)
    return r["mean_rankic"], r["rankicir"]

for name, cols in [("baseline49", BASELINE_COLS), ("full_ext", FULL_COLS)]:
    m = lgb.train(params, lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(test[cols], label=test["target_5"], feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    test = test.assign(s=m.predict(test[cols], num_iteration=m.best_iteration))
    ic1, ir1 = ev(test, "s", "target")
    ic5, ir5 = ev(test, "s", "target_5")
    print(f"{name} (n={len(cols)}) best_iter={m.best_iteration}")
    print(f"  2022 OOS: 1dIC={ic1:.4f} IR={ir1:.3f} | 5dIC={ic5:.4f} IR={ir5:.3f}")
