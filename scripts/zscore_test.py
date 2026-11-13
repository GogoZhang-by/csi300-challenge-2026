"""方向验证：截面z-score相对强度特征（双窗口门槛）+ Barra残差私榜风险"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from features import get_feature_columns
from train import lgb_params

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()
df["rank1"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

# 新增：核心特征的截面z-score（相对强度），只用当日截面
ZF = ["mom_5", "mom_20", "mom_60", "reversal_5", "ret_1", "vol_20",
      "change_ratio", "intraday_ret", "barra_动量"]
for c in ZF:
    mu = df.groupby("日期")[c].transform("mean")
    sd = df.groupby("日期")[c].transform("std").replace(0, np.nan)
    df[f"{c}_z"] = (df[c] - mu) / sd
zcols = [f"{c}_z" for c in ZF]

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]

def rankic(t, sc, lo, hi):
    ics = [spearmanr(g[sc], g["target"])[0] for d, g in t.groupby("日期") if lo <= d <= hi and len(g) >= 50]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics))

def window_test(tag, train_hi, test_lo, test_hi, nround=111):
    tr = clean.loc[clean["日期"].le(train_hi)]
    te = df.loc[df["日期"].between(test_lo, test_hi)].copy().sort_values(["代码", "日期"])
    out = {}
    for name, use_cols in [("baseline", cols), ("+zscore", cols + zcols)]:
        m = lgb.train(lgb_params(42),
            lgb.Dataset(tr[use_cols], label=tr["rank1"], feature_name=use_cols),
            num_boost_round=nround)
        te["sc"] = m.predict(te[use_cols].fillna(0))
        out[name] = rankic(te, "sc", test_lo, test_hi)
    print(f"{tag}: baseline={out['baseline']:.4f}  +z={out['+zscore']:.4f}  差={out['+zscore']-out['baseline']:+.4f}")
    return out["+zscore"] - out["baseline"]

d1 = window_test("2022全年     ", 20211231, 20220101, 20221231)
d2 = window_test("2021H2独立窗口", 20210630, 20210701, 20211231)
print(f"双窗口结论: {'通过(两窗都>=+0.002)' if d1 >= 0.002 and d2 >= 0.002 else '否决（不满足双窗口门槛）'}")

# Barra残差（w1）私榜风险快速检查
print("\n=== Barra残差（rank1 w1，2022）===")
BARRA = [f"barra_{n}" for n in ("市值","贝塔","动量","残差波动","非线性市值","账面市值比","流动性","盈利","成长","杠杆")]
tr = clean.loc[clean["日期"].le(20211231)]
te = df.loc[df["日期"].between(20220101, 20221231)].copy()
m = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr["rank1"], feature_name=cols), num_boost_round=111)
te["sc"] = m.predict(te[cols].fillna(0))
resid = pd.Series(np.nan, index=te.index)
for d, idx in te.groupby("日期").groups.items():
    g = te.loc[idx]
    X = np.column_stack([np.ones(len(g)), g[BARRA].fillna(0).values])
    b, *_ = np.linalg.lstsq(X, g["sc"].values, rcond=None)
    resid.loc[idx] = g["sc"].values - X @ b
te["scn"] = resid
print(f"原始score RankIC={rankic(te,'sc',20220101,20221231):.4f}  Barra残差 RankIC={rankic(te,'scn',20220101,20221231):.4f}")
