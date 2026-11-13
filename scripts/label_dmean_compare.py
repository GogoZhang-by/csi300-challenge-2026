"""对比4种组合的调仓日 RankIC（官方平台口径，rebalance=5, H=5）"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from train import lgb_params
from scipy.stats import spearmanr

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

from data_loader import DataLoader
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
md_s = md[["日期","代码","open"]].sort_values(["代码","日期"])
g = md_s.groupby("代码")["open"]
md_s["o1"] = g.shift(-1)
md_s["target_5_new"] = g.shift(-6)/md_s["o1"]-1.0
md_s["target_20_new"] = g.shift(-21)/md_s["o1"]-1.0
df = df.merge(md_s[["日期","代码","target_5_new","target_20_new"]], on=["日期","代码"], how="left")

# 四种标签（缓存已有 target/target_5/target_20，用新列避免冲突）
for h, col in [("1","target"),("5","target_5_new"),("20","target_20_new")]:
    df[f"rank{h}"] = df.groupby("日期")[col].rank(pct=True)-0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101,20221231)].copy()

# 调仓日：每5天取一个
all_dates = sorted(test["日期"].unique())
rebal_dates = all_dates[::5]

def rankic_rebal(score_col):
    ranks = []
    for d in rebal_dates:
        g = test.loc[test["日期"].eq(d)].dropna(subset=["target_5_new"])
        if len(g) < 50: continue
        ranks.append(spearmanr(g[score_col].values, g["target_5_new"].values)[0])
    r = pd.Series(ranks)
    return r.mean(), r.mean()/r.std()

results = {}
for label in ["rank1", "rank5", "rank20"]:
    m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train[label], feature_name=cols), num_boost_round=175)
    test["raw"] = m.predict(test[cols].fillna(0))
    test = test.sort_values(["代码","日期"])
    test["avg20"] = test.groupby("代码", sort=False)["raw"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    for sc_name in ["raw", "avg20"]:
        ric, icir = rankic_rebal(sc_name)
        results[f"{label}+{sc_name}"] = (ric, icir)
        print(f"{label}+{sc_name}: RankIC={ric:.4f} ICIR={icir:.3f}")

print("\n最优:", max(results.items(), key=lambda x: x[1][0]))
