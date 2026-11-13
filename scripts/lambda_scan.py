"""lambda_l2 精细扫描 + 双窗口稳健性验证"""
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

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]

def rankic_w(t, sc, lo, hi):
    ics = [spearmanr(g[sc], g["target"])[0] for d, g in t.groupby("日期") if lo <= d <= hi and len(g) >= 50]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics))

def ir_w(t, sc, lo, hi, cost=0.0015):
    dates = sorted(t.loc[t["日期"].between(lo, hi), "日期"].unique())[::5]
    prev, rs = None, []
    for d in dates:
        g = t.loc[t["日期"].eq(d)].dropna(subset=["target_5"])
        if len(g) < 100: continue
        hold = set(g.nlargest(50, sc)["代码"])
        to = 1 - len(hold & prev) / 50 if prev else 1.0
        ex = g.loc[g["代码"].isin(hold), "target_5"].mean() - g["target_5"].mean() - 2 * to * cost
        rs.append(ex); prev = hold
    r = pd.Series(rs)
    return r.mean() / r.std() * np.sqrt(242 / 5)

def test_lambda(l2, train_hi, test_lo, test_hi, label=""):
    tr = clean.loc[clean["日期"].le(train_hi)]
    seg = df.loc[df["日期"].between(test_lo - 100, test_hi)].copy().sort_values(["代码", "日期"])
    params = {**lgb_params(42), "lambda_l2": l2}
    m = lgb.train(params, lgb.Dataset(tr[cols], label=tr["rank1"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(clean.loc[clean["日期"].between(20220101,20221231), cols],
                                          label=clean.loc[clean["日期"].between(20220101,20221231), "rank1"],
                                          feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    seg["sc"] = m.predict(seg[cols].fillna(0))
    ric = rankic_w(seg, "sc", test_lo, test_hi)
    ir = ir_w(seg, "sc", test_lo, test_hi)
    print(f"  l2={l2:.1f} {label:14s} RankIC={ric:.4f} IR={ir:5.2f} 综合={0.7*ric+0.3*ir:.3f}")
    return ric, ir

print("=== lambda_l2 扫描（2022 OOS）===")
for l2 in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    test_lambda(l2, 20211231, 20220101, 20221231)

print("\n=== 双窗口稳健性：l2=1.0 vs baseline ===")
print("2022 全年:")
r0_22, i0_22 = test_lambda(0.0, 20211231, 20220101, 20221231, "baseline")
r1_22, i1_22 = test_lambda(1.0, 20211231, 20220101, 20221231, "l2=1.0")
print(f"  ΔRankIC={r1_22-r0_22:+.4f}  ΔIR={i1_22-i0_22:+.2f}")

print("2021H2 独立窗口:")
r0_21, i0_21 = test_lambda(0.0, 20210630, 20210701, 20211231, "baseline")
r1_21, i1_21 = test_lambda(1.0, 20210630, 20210701, 20211231, "l2=1.0")
print(f"  ΔRankIC={r1_21-r0_21:+.4f}  ΔIR={i1_21-i0_21:+.2f}")

print(f"\n双窗口结论: l2=1.0 {'通过（两窗RankIC和IR都不降）' if r1_22>=r0_22 and r1_21>=r0_21 and i1_22>=i0_22 and i1_21>=i0_21 else '需进一步验证'}")
