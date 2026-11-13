"""更长平均窗口 + top_n 敏感性"""
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

m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
              num_boost_round=140)

all_df = df.loc[df["日期"].between(20211101, 20221231)].copy()
all_df["raw_score"] = m.predict(all_df[cols].fillna(0))
all_df = all_df.sort_values(["代码", "日期"])
for w in [5, 10, 15, 20]:
    all_df[f"avg{w}"] = all_df.groupby("代码", sort=False)["raw_score"].transform(
        lambda s: s.rolling(w, min_periods=1).mean())
test2 = all_df.loc[all_df["日期"].between(20220101, 20221231)].copy()

def bt(test_df, score_col, top_n=50, cost=0.0015):
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

print("=== 平均窗口扫描 (top50) ===")
for w in [5, 10, 15, 20]:
    col = f"avg{w}"
    t = test2.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    ir, irn, to = bt(t, col)
    print(f"avg{w}: 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:.2f} 成本后IR={irn:.2f} 换手={to:.1%}")

print("\n=== top_n 扫描 (avg10 / avg20) ===")
for col in ["avg10", "avg20"]:
    for n in [50, 100, 150, 200]:
        ir, irn, to = bt(test2, col, top_n=n)
        print(f"{col} top{n}: 毛IR={ir:.2f} 成本后IR={irn:.2f} 换手={to:.1%}")
