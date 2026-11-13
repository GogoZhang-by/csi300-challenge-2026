"""逆向工程第1名：扫描高RankIC+高IR的可能配方"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from data_loader import DataLoader
from features import get_feature_columns, build_features
from train import lgb_params
from validate import RankICEvaluator

df = pd.read_parquet("result/feature_cache.parquet")
base_cols = get_feature_columns()

# 加回 rank 特征（截面 rank，需在 merge 后计算）
df = df.sort_values(["日期", "代码"]).reset_index(drop=True)
df["turn_rank"] = df.groupby("日期")["turn"].rank(pct=True)
df["vol_20_rank"] = df.groupby("日期")["vol_20"].rank(pct=True)
df["mom_5_rank"] = df.groupby("日期")["mom_5"].rank(pct=True)
rank_cols = ["turn_rank", "vol_20_rank", "mom_5_rank"]

# 构造 target_10 / target_20
sys.path.insert(0, "src")
from data_loader import DataLoader
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
md_s = md[["日期","代码","open"]].sort_values(["代码","日期"])
g = md_s.groupby("代码")["open"]
md_s["o1"] = g.shift(-1)
for h in [10, 20]:
    md_s[f"target_{h}"] = g.shift(-(h+1)) / md_s["o1"] - 1.0
df = df.merge(md_s[["日期","代码","target_10","target_20"]], on=["日期","代码"], how="left")

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

def run(tag, label, cols, n_iter=140, avg_window=1, top_n=50):
    tr = train.loc[train[label].notna() & train[label].abs().le(0.9)]
    m = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols),
                  num_boost_round=n_iter)
    t = test.assign(raw=m.predict(test[cols].fillna(0)))
    if avg_window > 1:
        t = t.sort_values(["代码", "日期"])
        t["s"] = t.groupby("代码", sort=False)["raw"].transform(lambda s: s.rolling(avg_window, min_periods=1).mean())
    else:
        t["s"] = t["raw"]
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s")
    ir, irn, to = backtest(t, "s", top_n=top_n)
    print(f"{tag:36s} 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 成本后IR={irn:5.2f} 换手={to:.1%}")

print("=== 标签周期扫描 (全特征, n_iter=140) ===")
for label in ["target", "target_5", "target_10", "target_20"]:
    run(f"{label} raw", label, base_cols, avg_window=1)
    run(f"{label} avg20", label, base_cols, avg_window=20)

print("\n=== 特征集扫描 (target_5) ===")
run("base49", "target_5", base_cols, avg_window=1)
run("+rank", "target_5", base_cols + rank_cols, avg_window=1)
run("-Barra", "target_5", [c for c in base_cols if not c.startswith("barra_")], avg_window=1)

print("\n=== 模型深度扫描 (target_5) ===")
for nl in [31, 63, 127, 255]:
    p = lgb_params(42); p["num_leaves"] = nl; p["max_depth"] = max(6, int(np.log2(nl)))
    tr = train.loc[train["target_5"].notna() & train["target_5"].abs().le(0.9)]
    m = lgb.train(p, lgb.Dataset(tr[base_cols], label=tr["target_5"], feature_name=base_cols), num_boost_round=140)
    t = test.assign(s=m.predict(test[base_cols].fillna(0)))
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s")
    ir, irn, to = backtest(t, "s")
    print(f"num_leaves={nl:3d} 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:5.2f} 成本后IR={irn:5.2f} 换手={to:.1%}")

print("\n=== 第1名可能路径: target_10/20 + 慢特征 + avg20 ===")
slow_ts = [c for c in base_cols if c not in ("ret_1", "mom_5", "max_ret_20", "intraday_ret")]
run("target_20 slow raw", "target_20", slow_ts, n_iter=200, avg_window=1)
run("target_20 slow avg20", "target_20", slow_ts, n_iter=200, avg_window=20, top_n=100)
run("target_10 slow avg20", "target_10", slow_ts, n_iter=200, avg_window=20, top_n=100)
