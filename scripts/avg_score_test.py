"""验证方案C: 多日预测平均 (score_t = mean_{k=0..4} model(features at t-k))
无未来函数: t-k 时刻的预测只用 ≤t-k 数据; 无跨日状态: 推理时在窗口内对5天各算一次特征。
"""
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
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
              num_boost_round=140)

# 全量打分（含2021年末，供 t-k 使用）
all_df = df.loc[df["日期"].between(20211101, 20221231)].copy()
all_df["raw_score"] = m.predict(all_df[cols].fillna(0))

# 多日预测平均: score_t = mean(raw_score_{t}, raw_score_{t-1}, ..., raw_score_{t-4})
all_df = all_df.sort_values(["代码", "日期"])
all_df["score_avg5"] = all_df.groupby("代码", sort=False)["raw_score"].transform(
    lambda s: s.rolling(5, min_periods=1).mean())
all_df["score_avg10"] = all_df.groupby("代码", sort=False)["raw_score"].transform(
    lambda s: s.rolling(10, min_periods=1).mean())

test2 = all_df.loc[all_df["日期"].between(20220101, 20221231)].copy()

def backtest_full(test_df, score_col, top_n=50, cost=0.0015):
    days = sorted(test_df["日期"].unique())
    rets, rets_net, turnovers = [], [], []
    prev_hold = None
    for d in days:
        g = test_df.loc[test_df["日期"].eq(d)].dropna(subset=["target"])
        if len(g) < 100: continue
        top = g.nlargest(top_n, score_col)
        hold = set(top["代码"])
        turnover = 1 - len(hold & prev_hold) / top_n if prev_hold is not None else 1.0
        turnovers.append(turnover)
        port = top["target"].mean()
        bench = g["target"].mean()
        rets.append(port - bench)
        rets_net.append(port - bench - 2 * turnover * cost)
        prev_hold = hold
    r, rn = pd.Series(rets), pd.Series(rets_net)
    ir = r.mean()/r.std()*np.sqrt(242) if r.std()>0 else np.nan
    irn = rn.mean()/rn.std()*np.sqrt(242) if rn.std()>0 else np.nan
    return ir, irn, np.mean(turnovers), r.mean()*242

print(f"{'方案':24s} {'1dIC':>7s} {'毛IR':>7s} {'成本后IR':>8s} {'日换手':>7s} {'毛年超额':>8s}")
for tag, col in [("单日预测", "raw_score"), ("5日预测平均", "score_avg5"), ("10日预测平均", "score_avg10")]:
    t = test2.dropna(subset=[col])
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col=col)
    ir, irn, to, ann = backtest_full(t, col)
    print(f"{tag:24s} {ric['mean_rankic']:7.4f} {ir:7.2f} {irn:8.2f} {to:7.1%} {ann:8.2%}")

# 不同 top_n
print("\n5日预测平均, 不同持仓数:")
for n in [30, 50, 80, 100]:
    ir, irn, to, ann = backtest_full(test2, "score_avg5", top_n=n)
    print(f"  top{n}: 毛IR={ir:.2f} 成本后IR={irn:.2f} 日换手={to:.1%}")
