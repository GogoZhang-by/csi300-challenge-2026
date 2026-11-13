"""验证当前提交模型(rank20+avg20)的净IR与换手，与上一版(rank5+avg20)对比"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from validate import RankICEvaluator

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

model = lgb.Booster(model_file="submission/model/lgb_reg.txt")

# avg20 需要前20交易日，从2021-12取数
seg = df.loc[df["日期"].between(20211201, 20221231)].copy()
seg["raw"] = model.predict(seg[cols].fillna(0))
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
print(f"当前提交(rank20+avg20): 1dIC={ric['mean_rankic']:.4f} 毛IR={ir:.2f} 净IR={irn:.2f} 换手={to:.1%} 公式分={0.7*ric['mean_rankic']+0.3*irn:.3f}")
