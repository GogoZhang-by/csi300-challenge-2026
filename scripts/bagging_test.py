"""模型 bagging：多种子平均 → 提升 RankIC（降低方差）"""
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

mask = (df["target"].notna() & df["target_5"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["target_5"].abs().le(0.60)
        & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

def backtest(test_df, score_col, top_n=50, cost=0.0015):
    rets, rets_net, tos = [], [], []
    prev = None
    for d in sorted(test_df["日期"].unique()):
        g = test_df.loc[test_df["日期"].eq(d)].dropna(subset=["target"])
        if len(g) < 100: continue
        hold = set(g.nlargest(top_n, score_col)["代码"])
        to = 1 - len(hold & prev) / top_n if prev is not None else 1.0
        tos.append(to)
        ex = g.nlargest(top_n, score_col)["target"].mean() - g["target"].mean()
        rets.append(ex); rets_net.append(ex - 2 * to * cost)
        prev = hold
    r, rn = pd.Series(rets), pd.Series(rets_net)
    return (r.mean()/r.std()*np.sqrt(242), rn.mean()/rn.std()*np.sqrt(242), np.mean(tos))

tr = train.loc[train["target_5"].notna() & train["target_5"].abs().le(0.9)]

# 单模型基准
m1 = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr["target_5"], feature_name=cols), num_boost_round=140)
test["s1"] = m1.predict(test[cols].fillna(0))

# Bagging: 5 个不同种子
preds = [test["s1"].values]
for seed in [7, 13, 21, 99]:
    m = lgb.train(lgb_params(seed), lgb.Dataset(tr[cols], label=tr["target_5"], feature_name=cols), num_boost_round=140)
    preds.append(m.predict(test[cols].fillna(0)))
test["s_bag5"] = np.mean(preds, axis=0)

# 更多种子
for seed in [3, 11, 27, 53, 77]:
    m = lgb.train(lgb_params(seed), lgb.Dataset(tr[cols], label=tr["target_5"], feature_name=cols), num_boost_round=140)
    preds.append(m.predict(test[cols].fillna(0)))
test["s_bag10"] = np.mean(preds, axis=0)

print(f"{'方案':12s} {'1dIC':>7s} {'毛IR':>7s} {'成本后IR':>8s} {'换手':>7s}")
for tag, col in [("单模型", "s1"), ("bag5", "s_bag5"), ("bag10", "s_bag10")]:
    t = test.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    ir, irn, to = backtest(t, col)
    print(f"{tag:12s} {ric['mean_rankic']:7.4f} {ir:7.2f} {irn:8.2f} {to:7.1%}")

# bag + avg20
print("\n=== bag + avg20 平滑 ===")
test = test.sort_values(["代码", "日期"])
for col in ["s1", "s_bag5", "s_bag10"]:
    tag = col.replace("s_", "")
    test[f"{col}_a20"] = test.groupby("代码", sort=False)[col].transform(lambda s: s.rolling(20, min_periods=1).mean())
    t = test.dropna(subset=[f"{col}_a20"])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=f"{col}_a20")
    ir, irn, to = backtest(t, f"{col}_a20")
    print(f"{tag:12s} {ric['mean_rankic']:7.4f} {ir:7.2f} {irn:8.2f} {to:7.1%}")
