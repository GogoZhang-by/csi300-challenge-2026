"""诚实OOS对比：模型只用≤2021训练，在2022上对比 score 后处理方案"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from data_loader import DataLoader
from features import get_feature_columns
from train import lgb_params
from validate import RankICEvaluator

dl = DataLoader("data_temp/Data")
ind = dl.load_hs300_industry()
ind_map = ind.sort_values("日期").groupby("代码")["行业代码"].last().to_dict()

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

mask = (df["target"].notna() & df["target_5"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["target_5"].abs().le(0.60)
        & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]

m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
              num_boost_round=140)  # 多种子实验的 best_iter 中位数

# 全量 2022（含停牌，不剔除——官方口径）
test = df.loc[df["日期"].between(20220101, 20221231)].copy()
test["score_raw"] = m.predict(test[cols].fillna(0))
test["industry"] = test["代码"].map(ind_map).fillna("UNKNOWN")

def lim(code, st, d):
    if st == 1: return 0.05
    if code.startswith("SH688"): return 0.20
    if code.startswith("SZ300") and d >= 20200824: return 0.20
    return 0.10
test["_lim"] = [lim(c, s, d) for c, s, d in zip(test["代码"], test["is_st"].fillna(0).astype(int), test["日期"])]
test["at_limit_up"] = (test["ret_1"] >= test["_lim"] - 0.002).astype(int)
test["at_limit_dn"] = (test["ret_1"] <= -test["_lim"] + 0.002).astype(int)

def backtest(df_, score_col, top_n=50):
    rets = []
    for d, g in df_.groupby("日期"):
        g = g.dropna(subset=["target"])
        if len(g) < 100: continue
        port = g.nlargest(top_n, score_col)["target"].mean()
        bench = g["target"].mean()
        rets.append(port - bench)
    r = pd.Series(rets)
    return (r.mean()/r.std()*np.sqrt(242) if r.std()>0 else np.nan), r.mean()*242

def zd(s): return (s - s.mean()) / (s.std() + 1e-12)

test["s_z"] = test.groupby("日期")["score_raw"].transform(zd)
test["s_rank"] = test.groupby("日期")["score_raw"].rank(pct=True)
test["s_ind"] = test["s_z"] - test.groupby(["日期","industry"])["s_z"].transform("mean")
test["s_ind_rank"] = test.groupby("日期")["s_ind"].rank(pct=True)
trd = test["is_suspend"]*5 + test["at_limit_up"]*2 + test["at_limit_dn"]*2
test["s_ind_trd"] = test["s_ind"] - trd
test["s_ind_trd_rank"] = test.groupby("日期")["s_ind_trd"].rank(pct=True)

print(f"{'方案':26s} {'IR':>7s} {'年化超额':>9s} {'1dRankIC':>9s} {'IR_IC口径':>9s}")
for name, col in [("V0_raw", "score_raw"), ("V1_rank输出", "s_rank"),
                  ("V2_行业中性", "s_ind"), ("V2b_行业中性+rank", "s_ind_rank"),
                  ("V4_行业中性+可交易+rank", "s_ind_trd_rank")]:
    ir, ann = backtest(test, col)
    ric = RankICEvaluator.evaluate(test.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    print(f"{name:26s} {ir:7.3f} {ann:9.2%} {ric['mean_rankic']:9.4f} {ric['rankicir']:9.3f}")
