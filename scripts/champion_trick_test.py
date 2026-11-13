"""验证Kaggle冠军核心trick：top特征的每日截面均值作为新特征"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from train import lgb_params
from validate import RankICEvaluator

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

# 找出与target_5最相关的top特征（训练期计算，避免泄露）
train_period = df.loc[df["日期"].le(20211231)]
corrs = train_period[cols + ["target_5"]].corr()["target_5"].abs().sort_values(ascending=False)
top_feats = [f for f in corrs.head(25).index if f not in ("target_5", "target", "target_20")][:20]
print("Top20相关特征:", top_feats)

# 每日截面均值（全期，无未来函数：只用当日截面）
for f in top_feats:
    df[f"{f}_dmean"] = df.groupby("日期")[f].transform("mean")

new_cols = cols + [f"{f}_dmean" for f in top_feats]

# 标签
df["rank20"] = df.groupby("日期")["target_20"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["rank20"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

def fit(label, cols_):
    tr = train.loc[train[label].notna()]
    return lgb.train(lgb_params(42), lgb.Dataset(tr[cols_], label=tr[label], feature_name=cols_),
                     num_boost_round=175)

for tag, cols_, label in [("baseline rank20", cols, "rank20"), ("+dmean rank20", new_cols, "rank20")]:
    m = fit(label, cols_)
    test["s"] = m.predict(test[cols_].fillna(0))
    test = test.sort_values(["代码", "日期"])
    test["sa20"] = test.groupby("代码", sort=False)["s"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    ric = RankICEvaluator.evaluate(test.rename(columns={"日期": "date"}), label_col="target", score_col="sa20")
    print(f"{tag}: 1dIC={ric['mean_rankic']:.4f} 正IC%={ric['positive_ratio']:.1%}")
