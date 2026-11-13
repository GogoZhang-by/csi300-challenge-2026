"""诊断3：严格复现 baseline 训练逻辑（无清洗、全量池、5日标签、原始49特征），评估1日/5日RankIC"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from validate import RankICEvaluator
from data_loader import DataLoader
from features import FACTOR_FEATURES, EXTRA_FACTOR_FEATURES, BARRA_FEATURES

# 全量并集特征（模仿 baseline 的 training_samples）
cache = "result/feature_cache_full.parquet"
full = pd.read_parquet(cache)
rel = [f"{n}_rel" for n in ("sma_20", "ema_20", "macd_diff_12_26_9",
                            "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14")]
full = full.rename(columns={n: f"barra_{n}" for n in BARRA_FEATURES})
BASELINE_COLS = list(FACTOR_FEATURES) + [f"barra_{n}" for n in BARRA_FEATURES] + list(EXTRA_FACTOR_FEATURES) + rel

# baseline 式清洗：停牌/缺价/非正价格
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
labels = dl.build_labels(md)
if "close" not in full.columns:
    full = full.merge(md[["日期", "代码", "close"]], on=["日期", "代码"], how="left")
for n in ("sma_20", "ema_20", "macd_diff_12_26_9", "macd_dea_12_26_9", "macd_hist_12_26_9", "atr_14"):
    if f"{n}_rel" not in full.columns:
        full[f"{n}_rel"] = full[n] / full["close"]
mask = (full["target"].notna() & full["target_5"].notna()
        & full["is_suspend"].eq(0) & full["close"].gt(0))
clean = full.loc[mask].copy()

# 注意：baseline 的 training_samples 中 holding_period=5，label 是5日
# 但它训练时用的是 exit_date ≤ train_end 的样本，不是日期本身
# 这里近似：按日期切分，但 label 用 5 日
# baseline 的 train_end=2021-12-31，val=2022 全年
train = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].gt(20211231) & clean["日期"].le(20221231)]
test = clean.loc[clean["日期"].gt(20211231)]  # same as val in this case

params = {"objective": "regression", "metric": "l2", "verbosity": -1,
          "learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
          "min_data_in_leaf": 100, "feature_fraction": 0.85,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "seed": 42, "num_threads": 8, "deterministic": True, "force_col_wise": True}

def ev(df_, col, label="target"):
    return RankICEvaluator.evaluate(df_.rename(columns={"日期": "date"}), label_col=label, score_col=col)["mean_rankic"]

for lab in ["target_5", "target"]:
    m = lgb.train(params, lgb.Dataset(train[BASELINE_COLS], label=train[lab], feature_name=BASELINE_COLS),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(val[BASELINE_COLS], label=val[lab], feature_name=BASELINE_COLS)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    val = val.assign(s=m.predict(val[BASELINE_COLS], num_iteration=m.best_iteration))
    # 只评估 Barra 成分股
    bar = pd.read_parquet("result/feature_cache.parquet")
    bar_codes = set(bar["代码"].unique())
    val_b = val[val["代码"].isin(bar_codes)]
    print(f"lab={lab} best_iter={m.best_iteration}")
    print(f"  val Barra: 1dIC={ev(val_b,'s','target'):.4f}  5dIC={ev(val_b,'s','target_5'):.4f}")

# 再试试：用冻结 baseline 模型在 2022 年上评估1日 vs 5日
print("\n--- 冻结 baseline 模型 ---")
from inference import InferenceModel as BaseInf
bm = BaseInf("baseline_temp/baseline_submission 2/model")
md22 = md.loc[md["日期"].between(20220101, 20221231)]
f33_22 = dl.load_factor33(end_date=20221231)
bar22 = dl.load_barra(end_date=20221231)

import numpy as np
from scipy.stats import spearmanr
vals1, vals5 = [], []
for d in sorted(bar22["日期"].unique()):
    data = {"MarketData": md22.loc[md22["日期"].eq(d)],
            "Factor33_winsor": f33_22.loc[f33_22["日期"].eq(d)],
            "Barra_factor": bar22.loc[bar22["日期"].eq(d)]}
    pred = bm.predict(f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}", data)
    pred["_日期"] = pred["date"].str.replace("-","").astype(int)
    merged = pred.merge(labels, left_on=["_日期","code"], right_on=["日期","代码"], how="left")
    merged = merged.dropna(subset=["target", "target_5"])
    if len(merged) < 10: continue
    ic1, _ = spearmanr(merged["score"], merged["target"])
    ic5, _ = spearmanr(merged["score"], merged["target_5"])
    vals1.append(ic1); vals5.append(ic5)
print(f"baseline 2022 冻结模型: 1dIC={np.mean(vals1):.4f}  5dIC={np.mean(vals5):.4f}")
