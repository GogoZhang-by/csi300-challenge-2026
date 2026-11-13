"""P0修正网格：标签{rank1,rank5} x 平滑窗口{1,5,10,20} + Barra中性化
诚实OOS: train<=2021, test=2022。指标: 日频RankIC@1(榜单口径) + 5日调仓净IR代理
"""
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
df["rank5"] = df.groupby("日期")["target_5"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]

# 2021-12 垫窗 + 2022 测试
seg = df.loc[df["日期"].between(20211201, 20221231)].copy().sort_values(["代码", "日期"])
test_dates = sorted(seg.loc[seg["日期"].between(20220101, 20221231), "日期"].unique())
rebal_dates = test_dates[::5]

BARRA = [f"barra_{n}" for n in ("市值","贝塔","动量","残差波动","非线性市值","账面市值比","流动性","盈利","成长","杠杆")]

def metrics(sc_col, t):
    t = t.dropna(subset=[sc_col])
    ics = []
    for d, g in t.groupby("日期"):
        if d < 20220101 or len(g) < 50:
            continue
        v = spearmanr(g[sc_col], g["target"])[0]
        if np.isfinite(v):
            ics.append(v)
    ric = float(np.mean(ics))
    prev, rs = None, []
    for d in rebal_dates:
        g = t.loc[t["日期"].eq(d)].dropna(subset=["target_5"])
        if len(g) < 100:
            continue
        hold = set(g.nlargest(50, sc_col)["代码"])
        to = 1 - len(hold & prev) / 50 if prev else 1.0
        ex = g.loc[g["代码"].isin(hold), "target_5"].mean() - g["target_5"].mean() - 2 * to * 0.0015
        rs.append(ex)
        prev = hold
    r = pd.Series(rs)
    ir = r.mean() / r.std() * np.sqrt(242 / 5)
    return ric, ir

def neutralize(t, sc_col):
    out = pd.Series(np.nan, index=t.index)
    for d, idx in t.groupby("日期").groups.items():
        g = t.loc[idx]
        X = g[BARRA].fillna(0).values
        X = np.column_stack([np.ones(len(X)), X])
        y = g[sc_col].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[idx] = y - X @ beta
    return out

results = {}
models = {}
for label in ["rank1", "rank5"]:
    tr = train.loc[train[label].notna()]
    m = lgb.train(lgb_params(42), lgb.Dataset(tr[cols], label=tr[label], feature_name=cols),
                  num_boost_round=175)
    models[label] = m
    seg["raw"] = m.predict(seg[cols].fillna(0))
    for w in [1, 5, 10, 20]:
        if w == 1:
            seg["sc"] = seg["raw"]
        else:
            seg["sc"] = seg.groupby("代码", sort=False)["raw"].transform(
                lambda s: s.rolling(w, min_periods=1).mean())
        ric, ir = metrics("sc", seg)
        results[f"{label}_w{w}"] = (ric, ir)
        print(f"{label} w={w:2d}: RankIC@1={ric:.4f} 净IR={ir:5.2f} 综合={0.7*ric+0.3*ir:.3f}")
    # Barra 中性化（w=5）
    seg["sc"] = seg.groupby("代码", sort=False)["raw"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    seg["scn"] = neutralize(seg, "sc")
    ric, ir = metrics("scn", seg)
    results[f"{label}_w5_barra残差"] = (ric, ir)
    print(f"{label} w=5 +Barra残差: RankIC@1={ric:.4f} 净IR={ir:5.2f} 综合={0.7*ric+0.3*ir:.3f}")

# 双标签融合（zscore后等权, w=5）
seg["s1"] = models["rank1"].predict(seg[cols].fillna(0))
seg["s5"] = models["rank5"].predict(seg[cols].fillna(0))
for c in ["s1", "s5"]:
    seg[c] = seg.groupby("代码", sort=False)[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
    seg[c + "z"] = seg.groupby("日期")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
seg["mix"] = 0.5 * seg["s1z"] + 0.5 * seg["s5z"]
ric, ir = metrics("mix", seg)
results["rank1+rank5融合_w5"] = (ric, ir)
print(f"rank1+rank5融合 w=5: RankIC@1={ric:.4f} 净IR={ir:5.2f} 综合={0.7*ric+0.3*ir:.3f}")

print("\n=== 综合分排序 ===")
for k, (ric, ir) in sorted(results.items(), key=lambda x: -(0.7 * x[1][0] + 0.3 * x[1][1])):
    print(f"{k:22s} RankIC@1={ric:.4f} 净IR={ir:5.2f} 综合={0.7*ric+0.3*ir:.3f}")
