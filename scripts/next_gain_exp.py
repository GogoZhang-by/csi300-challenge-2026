"""下一短板(IR)提分实验：在 target_5_rank + avg20 基础上扫描候选 trick
全部诚实 OOS：train≤2021, test=2022
"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from train import lgb_params
from data_loader import DataLoader
from validate import RankICEvaluator

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

# 多周期 rank 标签
for h in [5, 10, 20]:
    col = f"target_{h}" if h != 5 else "target_5"
    if col not in df.columns and h != 5:
        dl = DataLoader("data_temp/Data")
        md = dl.load_market_data()[["日期", "代码", "open"]].sort_values(["代码", "日期"])
        g = md.groupby("代码")["open"]
        md["o1"] = g.shift(-1)
        md[f"target_{h}"] = g.shift(-(h + 1)) / md["o1"] - 1.0
        df = df.merge(md[["日期", "代码", f"target_{h}"]], on=["日期", "代码"], how="left")
    df[f"rank{h}"] = df.groupby("日期")[f"target_{h}" if h != 1 else "target"].rank(pct=True) - 0.5
df["rank1"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

# 涨停/停牌标记
def lim(code, st, d):
    if st == 1: return 0.05
    if code.startswith("SH688"): return 0.20
    if code.startswith("SZ300") and d >= 20200824: return 0.20
    return 0.10
test["_lim"] = [lim(c, s, d) for c, s, d in zip(test["代码"], test["is_st"].fillna(0).astype(int), test["日期"])]
test["at_up"] = (test["ret_1"] >= test["_lim"] - 0.002).astype(int)
test["at_dn"] = (test["ret_1"] <= -test["_lim"] + 0.002).astype(int)

def fit(label):
    tr = train.loc[train[label].notna()]
    return lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols),
                     num_boost_round=175)

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

def report(tag, t, col):
    t = t.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期": "date"}), label_col="target", score_col=col)
    ir, irn, to = bt(t, col)
    sc = 0.7 * ric["mean_rankic"] + 0.3 * irn
    print(f"{tag:30s} 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 净IR={irn:5.2f} 换手={to:5.1%} 公式分={sc:.3f}")

# 三个 rank 周期模型
preds = {}
for h in [1, 5, 10, 20]:
    preds[h] = fit(f"rank{h}").predict(test[cols].fillna(0))
    test[f"p{h}"] = preds[h]

test = test.sort_values(["代码", "日期"])
print("=== A. rank标签周期 × avg窗口 ===")
for h in [1, 5, 10, 20]:
    for w in [10, 20, 30, 40]:
        test[f"p{h}a{w}"] = test.groupby("代码", sort=False)[f"p{h}"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        report(f"rank{h} avg{w}", test, f"p{h}a{w}")

print("\n=== B. rank5 avg20 + 可交易性惩罚 ===")
for pen_up in [0.0, 0.5, 1.0]:
    for pen_susp in [0.0, 1.0]:
        test["adj"] = test["p5a20"] - pen_up * (test["at_up"] + test["at_dn"]) - pen_susp * test["is_suspend"].fillna(0)
        report(f"pen_up={pen_up} susp={pen_susp}", test, "adj")

print("\n=== C. 截面 rank 输出（替代原始分值） ===")
test["p5a20_rank"] = test.groupby("日期")["p5a20"].rank(pct=True)
report("rank5 avg20 ->截面rank", test, "p5a20_rank")

print("\n=== D. 双周期等权融合 + avg20 ===")
test["mix"] = 0.5 * test["p5"] + 0.5 * test["p10"]
test["mix_a20"] = test.groupby("代码", sort=False)["mix"].transform(lambda s: s.rolling(20, min_periods=1).mean())
report("0.5rank5+0.5rank10 avg20", test, "mix_a20")
