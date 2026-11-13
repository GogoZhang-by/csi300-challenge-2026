"""换手率诊断：不同标签周期/特征速度下，top50组合的换手与成本后IR"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns, TS_FEATURES
from train import lgb_params
from validate import RankICEvaluator

df = pd.read_parquet("result/feature_cache.parquet")
cols_all = get_feature_columns()

# 构造 target_10/target_20
sys.path.insert(0, "src")
from data_loader import DataLoader
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
md_s = md[["日期","代码","open"]].sort_values(["代码","日期"])
g = md_s.groupby("代码")["open"]
md_s["o1"] = g.shift(-1)
for h in [10, 20]:
    md_s[f"o{h+1}"] = g.shift(-(h+1))
    md_s[f"target_{h}"] = md_s[f"o{h+1}"] / md_s["o1"] - 1.0
df = df.merge(md_s[["日期","代码","target_10","target_20"]], on=["日期","代码"], how="left")

SLOW_TS = ("mom_20", "mom_60", "vol_20", "vol_60", "overnight_vol_20", "hl_range_20", "amt_zscore_120")
cols_slow = [c for c in cols_all if c not in ("ret_1", "mom_5", "max_ret_20", "intraday_ret")]

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

def backtest_full(test_df, score_col, top_n=50, cost=0.0015):
    """返回 (IR无成本, IR扣成本, 日均单边换手)"""
    days = sorted(test_df["日期"].unique())
    rets, rets_net, turnovers = [], [], []
    prev_hold = None
    for d in days:
        g = test_df.loc[test_df["日期"].eq(d)].dropna(subset=["target"])
        if len(g) < 100: continue
        top = g.nlargest(top_n, score_col)
        hold = set(top["代码"])
        if prev_hold is not None:
            turnover = 1 - len(hold & prev_hold) / top_n
        else:
            turnover = 1.0
        turnovers.append(turnover)
        port = top["target"].mean()
        bench = g["target"].mean()
        rets.append(port - bench)
        rets_net.append(port - bench - 2 * turnover * cost)  # 双边
        prev_hold = hold
    r, rn = pd.Series(rets), pd.Series(rets_net)
    ir = r.mean()/r.std()*np.sqrt(242) if r.std()>0 else np.nan
    irn = rn.mean()/rn.std()*np.sqrt(242) if rn.std()>0 else np.nan
    return ir, irn, np.mean(turnovers)

def run(tag, train_df, label, cols, n_iter):
    tr = train_df.loc[train_df[label].notna() & train_df[label].abs().le(0.9)]
    m = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols),
                  num_boost_round=n_iter)
    t = test.assign(s=m.predict(test[cols].fillna(0)))
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s")
    ir, irn, to = backtest_full(t, "s")
    print(f"{tag:28s} 1dIC={ric['mean_rankic']:.4f} | IR={ir:6.2f} | IR(成本后)={irn:6.2f} | 日换手={to:.1%}")

print(f"{'方案':28s} 成本=单边0.15%")
run("当前: 5日标签+全特征", train, "target_5", cols_all, 114)
run("1日标签+全特征", train, "target", cols_all, 114)
run("10日标签+全特征", train, "target_10", cols_all, 114)
run("20日标签+全特征", train, "target_20", cols_all, 114)
run("5日标签+慢特征", train, "target_5", cols_slow, 114)
run("10日标签+慢特征", train, "target_10", cols_slow, 114)
