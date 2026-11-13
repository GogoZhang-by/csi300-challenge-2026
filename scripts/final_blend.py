"""最终方案对比：target_5_rank + target_rank 双模型融合 + avg 平滑"""
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
df["target_5_rank"] = df.groupby("日期")["target_5"].rank(pct=True) - 0.5
df["target_rank"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

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

def train_model(label, n=140):
    tr = train.loc[train[label].notna()]
    return lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols), num_boost_round=n)

m5 = train_model("target_5_rank")
m1 = train_model("target_rank")
test["p5"] = m5.predict(test[cols].fillna(0))
test["p1"] = m1.predict(test[cols].fillna(0))
test["blend"] = 0.5 * test["p5"] + 0.5 * test["p1"]

test = test.sort_values(["代码", "日期"])
for src in ["p5", "p1", "blend"]:
    for w in [10, 20]:
        test[f"{src}_a{w}"] = test.groupby("代码", sort=False)[src].transform(
            lambda s: s.rolling(w, min_periods=1).mean())

print(f"{'方案':22s} {'1dIC':>7s} {'毛IR':>6s} {'成本后IR':>8s} {'换手':>6s} {'公式分':>7s}")
for tag, col in [("p5(5日rank) avg20", "p5_a20"), ("blend avg10", "blend_a10"),
                 ("blend avg20", "blend_a20"), ("p1(1日rank) avg10", "p1_a10")]:
    t = test.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    ir, irn, to = backtest(t, col)
    score = 0.7 * ric["mean_rankic"] + 0.3 * irn
    print(f"{tag:22s} {ric['mean_rankic']:7.4f} {ir:6.2f} {irn:8.2f} {to:6.1%} {score:7.3f}")
