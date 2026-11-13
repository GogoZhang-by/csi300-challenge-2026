"""本地 IR 近似回测：对比 score 后处理方案对 IR 的影响
组合: 每日 score top-50 等权, 收益 = O(t+1)→O(t+2)
基准: 当日全池300只等权 O2O (近似沪深300)
IR = mean(超额)/std(超额) * sqrt(242)
"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from pathlib import Path
from data_loader import DataLoader
from features import build_features, get_feature_columns

dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
f33 = dl.load_factor33()
bar = dl.load_barra()
labels = dl.build_labels(md)

# 行业映射（取每只股票最新行业分类，固化用）
ind = dl.load_hs300_industry()
ind_map = ind.sort_values("日期").groupby("代码")["行业代码"].last().to_dict()

# 构建特征并预测（用 submission 模型，2022年）
df = pd.read_parquet("result/feature_cache.parquet")
meta = json.load(open("submission/model/metadata.json", encoding="utf-8"))
cols = meta["feature_columns"]
model = lgb.Booster(model_file="submission/model/lgb_reg.txt")

test = df.loc[df["日期"].between(20220101, 20221231)].copy()
test["score_raw"] = model.predict(test[cols].fillna(0))
test["industry"] = test["代码"].map(ind_map).fillna("UNKNOWN")

# limit 阈值推断（与 features.py 一致逻辑，用 is_st + 代码前缀）
def lim(code, st, d):
    if st == 1: return 0.05
    if code.startswith("SH688"): return 0.20
    if code.startswith("SZ300") and d >= 20200824: return 0.20
    return 0.10
test["_lim"] = [lim(c, s, d) for c, s, d in zip(test["代码"], test["is_st"].fillna(0).astype(int), test["日期"])]
test["at_limit_up"] = (test["ret_1"] >= test["_lim"] - 0.002).astype(int)
test["at_limit_dn"] = (test["ret_1"] <= -test["_lim"] + 0.002).astype(int)


def backtest(df_, score_col, top_n=50):
    """每日 top_n 等权 vs 全池等权，返回 IR"""
    rets = []
    for d, g in df_.groupby("日期"):
        g = g.dropna(subset=["target"])
        if len(g) < 100:
            continue
        top = g.nlargest(top_n, score_col)
        port = top["target"].mean()
        bench = g["target"].mean()
        rets.append(port - bench)
    r = pd.Series(rets)
    ir = r.mean() / r.std() * np.sqrt(242) if r.std() > 0 else np.nan
    return ir, r.mean() * 242


def zscore_daily(s):
    return (s - s.mean()) / (s.std() + 1e-12)


variants = {}

# V0: 原始 score
variants["V0_原始score"] = test.assign(s=test["score_raw"])

# V1: 截面 zscore
test["s_z"] = test.groupby("日期")["score_raw"].transform(zscore_daily)
variants["V1_zscore"] = test.assign(s=test["s_z"])

# V2: 行业中性化（zscore 后按行业去均值）
test["s_ind"] = test["s_z"] - test.groupby(["日期", "industry"])["s_z"].transform("mean")
variants["V2_行业中性"] = test.assign(s=test["s_ind"])

# V3: 可交易性惩罚（停牌→垫底，涨停→-2σ，跌停→-2σ）
adj = test["s_z"].copy()
adj = adj - test["is_suspend"] * 5 - test["at_limit_up"] * 2 - test["at_limit_dn"] * 2
test["s_trd"] = adj
variants["V3_可交易惩罚"] = test.assign(s=test["s_trd"])

# V4: 行业中性 + 可交易
adj2 = test["s_ind"].copy()
adj2 = adj2 - test["is_suspend"] * 5 - test["at_limit_up"] * 2 - test["at_limit_dn"] * 2
test["s_all"] = adj2
variants["V4_行业中性+可交易"] = test.assign(s=test["s_all"])

from validate import RankICEvaluator
print(f"{'方案':22s} {'IR':>7s} {'年化超额':>9s} {'1dRankIC':>9s}")
for name, v in variants.items():
    ir, ann = backtest(v, "s")
    ric = RankICEvaluator.evaluate(v.rename(columns={"日期": "date"}), label_col="target", score_col="s")
    print(f"{name:22s} {ir:7.3f} {ann:9.2%} {ric['mean_rankic']:9.4f}")
