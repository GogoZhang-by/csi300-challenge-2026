"""诚实OOS复核：≤2021训练 + 2022早停(复刻交付流程, 不refit) + avg20 净IR"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from train import lgb_params
from data_loader import DataLoader
from validate import RankICEvaluator

dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
mds = md[["日期", "代码", "open"]].sort_values(["代码", "日期"])
g = mds.groupby("代码")["open"]
mds["o1"] = g.shift(-1)
mds["target"] = g.shift(-2) / mds["o1"] - 1.0
mds["target_20"] = g.shift(-21) / mds["o1"] - 1.0

df = pd.read_parquet("result/feature_cache.parquet").drop(columns=["target", "target_5"], errors="ignore")
df = df.merge(mds[["日期", "代码", "target", "target_20"]], on=["日期", "代码"], how="left")
df["rank20"] = df.groupby("日期")["target_20"].rank(pct=True) - 0.5
cols = get_feature_columns()

mask = (df["target"].notna() & df["rank20"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]

m = lgb.train(
    lgb_params(42),
    lgb.Dataset(train[cols], label=train["rank20"], feature_name=cols),
    num_boost_round=500,
    valid_sets=[lgb.Dataset(val[cols], label=val["rank20"], feature_name=cols)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
print(f"OOS early-stop best_iter = {m.best_iteration} (交付模型=109)")

# 全量2022含停牌打分（官方全池口径），avg20 需2021年末垫窗
seg = df.loc[df["日期"].between(20211201, 20221231)].copy()
seg["raw"] = m.predict(seg[cols].fillna(0))
seg = seg.sort_values(["代码", "日期"])
seg["s"] = seg.groupby("代码", sort=False)["raw"].transform(lambda s: s.rolling(20, min_periods=1).mean())
test = seg.loc[seg["日期"].between(20220101, 20221231)]

def bt(test_df, score_col, top_n=50, cost=0.0015):
    rets, rets_n, tos = [], [], []
    prev = None
    for d in sorted(test_df["日期"].unique()):
        g = test_df.loc[test_df["日期"].eq(d)].dropna(subset=["target"])
        if len(g) < 100: continue
        hold = set(g.nlargest(top_n, score_col)["代码"])
        to = 1 - len(hold & prev) / top_n if prev is not None else 1.0
        tos.append(to)
        ex = g.nlargest(top_n, score_col)["target"].mean() - g["target"].mean()
        rets.append(ex); rets_n.append(ex - 2 * to * cost)
        prev = hold
    r, rn = pd.Series(rets), pd.Series(rets_n)
    return r.mean()/r.std()*np.sqrt(242), rn.mean()/rn.std()*np.sqrt(242), np.mean(tos)

ric = RankICEvaluator.evaluate(test.rename(columns={"日期": "date"}), label_col="target", score_col="s")
ir, irn, to = bt(test, "s")
print(f"诚实OOS rank20+avg20: 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:.2f} 净IR={irn:.2f} 换手={to:.1%}")
print(f"公式分(本地近似) = {0.7*ric['mean_rankic']+0.3*irn:.3f}  | 上一版 rank5+avg20 = 0.0311 / 1.35 / 0.428")
