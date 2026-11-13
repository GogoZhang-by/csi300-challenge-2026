"""在反解参数下，测试模型正则化对IR的影响（不碰特征和标签）"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from features import get_feature_columns
from train import lgb_params

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()
df["rank1"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]

seg = df.loc[df["日期"].between(20211201, 20221231)].copy().sort_values(["代码", "日期"])

def rankic_daily(t, sc):
    ics = [spearmanr(g[sc], g["target"])[0] for d, g in t.groupby("日期")
           if d >= 20220101 and len(g) >= 50]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics))

def ir_proxy(t, sc, cost=0.0015):
    rebal = sorted(t.loc[t["日期"].between(20220101, 20221231), "日期"].unique())[::5]
    prev, rs, tos = None, [], []
    for d in rebal:
        g = t.loc[t["日期"].eq(d)].dropna(subset=["target_5"])
        if len(g) < 100: continue
        hold = set(g.nlargest(50, sc)["代码"])
        to = 1 - len(hold & prev) / 50 if prev else 1.0
        tos.append(to)
        ex = g.loc[g["代码"].isin(hold), "target_5"].mean() - g["target_5"].mean() - 2 * to * cost
        rs.append(ex); prev = hold
    r = pd.Series(rs)
    return r.mean() / r.std() * np.sqrt(242 / 5), float(np.mean(tos))

# 正则化扫描
variants = {
    "baseline (num_leaves=31)": lgb_params(42),
    "num_leaves=15":          {**lgb_params(42), "num_leaves": 15},
    "num_leaves=7":           {**lgb_params(42), "num_leaves": 7},
    "min_child_samples=500":  {**lgb_params(42), "min_child_samples": 500},
    "min_child_samples=1000": {**lgb_params(42), "min_child_samples": 1000},
    "lambda_l2=1.0":          {**lgb_params(42), "lambda_l2": 1.0},
    "lambda_l2=5.0":          {**lgb_params(42), "lambda_l2": 5.0},
    "leaves15+min500":        {**lgb_params(42), "num_leaves": 15, "min_child_samples": 500},
}

print(f"{'配置':28s} {'RankIC':>8s} {'IR代理':>8s} {'换手':>7s} {'综合':>8s}")
print("-" * 62)
baseline_ric = baseline_ir = None
for name, params in variants.items():
    m = lgb.train(params, lgb.Dataset(train[cols], label=train["rank1"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(clean.loc[clean["日期"].between(20220101,20221231), cols],
                                          label=clean.loc[clean["日期"].between(20220101,20221231), "rank1"],
                                          feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    seg["sc"] = m.predict(seg[cols].fillna(0))
    ric = rankic_daily(seg, "sc")
    ir, to = ir_proxy(seg, "sc")
    score = 0.7 * ric + 0.3 * ir
    if name.startswith("baseline"):
        baseline_ric, baseline_ir = ric, ir
    print(f"{name:28s} {ric:8.4f} {ir:8.2f} {to:6.1%} {score:8.3f}")

print(f"\n基准: RankIC={baseline_ric:.4f} IR={baseline_ir:.2f}")
print("门槛: RankIC不低于基准-0.001 且 IR不低于基准")
