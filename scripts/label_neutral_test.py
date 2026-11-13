"""标签截面中性化测试：减去每日横截面均值/用截面rank作为标签"""
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

# 截面中性化标签
df["target_5_demean"] = df["target_5"] - df.groupby("日期")["target_5"].transform("mean")
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

def run(tag, label, n_iter=140):
    tr = train.loc[train[label].notna() & train[label].abs().le(0.9)]
    m = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols), num_boost_round=n_iter)
    t = test.assign(s=m.predict(test[cols].fillna(0)))
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s")
    ir, irn, to = backtest(t, "s")
    print(f"{tag:26s} 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 成本后IR={irn:5.2f} 换手={to:.1%}")
    return m, t

print("=== 标签中性化扫描 ===")
run("target_5 (基准)", "target_5")
run("target_5_demean", "target_5_demean")
run("target_5_rank", "target_5_rank")
m, t_rank = run("target_rank", "target_rank")

# 组合: target_rank + avg20 平滑
print("\n=== target_rank + avg20 平滑 ===")
t_rank = t_rank.sort_values(["代码", "日期"])
t_rank["s_a20"] = t_rank.groupby("代码", sort=False)["s"].transform(lambda s: s.rolling(20, min_periods=1).mean())
t_rank["s_a10"] = t_rank.groupby("代码", sort=False)["s"].transform(lambda s: s.rolling(10, min_periods=1).mean())
t_rank["s_a5"] = t_rank.groupby("代码", sort=False)["s"].transform(lambda s: s.rolling(5, min_periods=1).mean())
for w in [5, 10, 20]:
    col = f"s_a{w}"
    t = t_rank.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    ir, irn, to = backtest(t, col)
    print(f"target_rank avg{w:2d}  1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 成本后IR={irn:5.2f} 换手={to:.1%}")

# target_5_rank + avg20
print("\n=== target_5_rank + avg20 ===")
tr2 = train.loc[train["target_5_rank"].notna()]
m2 = lgb.train(lgb_params(42), lgb.Dataset(tr2[cols], label=tr2["target_5_rank"], feature_name=cols),
               num_boost_round=140)
test2 = test.assign(s=m2.predict(test[cols].fillna(0))).sort_values(["代码","日期"])
test2["s_a20"] = test2.groupby("代码", sort=False)["s"].transform(lambda s: s.rolling(20, min_periods=1).mean())
t = test2.dropna(subset=["s_a20"])
ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s_a20")
ir, irn, to = backtest(t, "s_a20")
print(f"target_5_rank avg20 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 成本后IR={irn:5.2f} 换手={to:.1%}")
