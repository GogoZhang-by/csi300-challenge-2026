"""验证：时序特征5日滚动平滑 → 降换手 → 成本后IR转正"""
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

# 对全部时序特征做5日滚动均值（按代码分组，组内rolling —— 推理时用120日窗口可复现）
df = df.sort_values(["代码", "日期"]).reset_index(drop=True)
for c in TS_FEATURES:
    df[f"{c}_s5"] = df.groupby("代码", sort=False)[c].transform(lambda s: s.rolling(5, min_periods=3).mean())

cols_smooth = [c for c in cols_all if c not in TS_FEATURES] + [f"{c}_s5" for c in TS_FEATURES]

mask = (df["target"].notna() & df["target_5"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["target_5"].abs().le(0.60)
        & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
test = df.loc[df["日期"].between(20220101, 20221231)].copy()

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
    return ir, irn, np.mean(turnovers)

def run(tag, cols, n_iter=140):
    m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
                  num_boost_round=n_iter)
    t = test.assign(s=m.predict(test[cols].fillna(0)))
    ric = RankICEvaluator.evaluate(t.rename(columns={"日期":"date"}), label_col="target", score_col="s")
    ir, irn, to = backtest_full(t, "s")
    print(f"{tag:32s} 1dIC={ric['mean_rankic']:.4f} | 毛IR={ir:6.2f} | 成本后IR={irn:6.2f} | 日换手={to:.1%}")

run("当前: 原始时序特征", cols_all)
run("平滑: 时序特征5日均值", cols_smooth)

# 组合：原始+平滑同时给模型
cols_both = cols_all + [f"{c}_s5" for c in TS_FEATURES]
run("原始+平滑并存", cols_both)
