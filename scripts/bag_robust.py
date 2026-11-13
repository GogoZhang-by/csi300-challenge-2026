"""bagging稳健性：5种子、2022 H1/H2分段、不同OOS窗口(2021H2)；另测ICIR feval"""
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
train = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]
tr1 = train.loc[train["rank1"].notna()]

def rankic_window(t, sc, lo, hi, col_target="target"):
    ics = [spearmanr(g[sc], g[col_target])[0] for d, g in t.groupby("日期")
           if lo <= d <= hi and len(g) >= 50]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)), float(np.mean(ics) / np.std(ics))

# 用冠军迭代数 111
SEEDS = [42, 7, 2024, 314, 99]
preds = {}
for sd in SEEDS:
    mm = lgb.train(lgb_params(sd),
        lgb.Dataset(tr1[cols], label=tr1["rank1"], feature_name=cols),
        num_boost_round=111)
    preds[sd] = mm

def predict_all(t):
    return {sd: m.predict(t[cols].fillna(0)) for sd, m in preds.items()}

def report(tag, seg, sc, lo=20220101, hi=20221231):
    ric, icir = rankic_window(seg, sc, lo, hi)
    ric_h1, _ = rankic_window(seg, sc, 20220101, 20220630)
    ric_h2, _ = rankic_window(seg, sc, 20220701, 20221231)
    print(f"{tag:26s} 全年={ric:.4f}(ICIR {icir:.3f})  H1={ric_h1:.4f}  H2={ric_h2:.4f}")
    return ric

# 2022
seg22 = df.loc[df["日期"].between(20211201, 20221231)].copy().sort_values(["代码", "日期"])
pp = predict_all(seg22)
seg22["p42"] = pp[42]
seg22["bag3"] = np.mean([pp[s] for s in [42, 7, 2024]], axis=0)
seg22["bag5"] = np.mean([pp[s] for s in SEEDS], axis=0)
print("=== 2022 ===")
report("单模型 seed42", seg22, "p42")
report("3种子bag", seg22, "bag3")
report("5种子bag", seg22, "bag5")

# 2021H2 独立窗口（训练<=2021H1）
print("\n=== 2021H2 独立窗口（train<=20210630）===")
tr_21 = clean.loc[clean["日期"].le(20210630)]
seg21 = df.loc[df["日期"].between(20210701, 20211231)].copy().sort_values(["代码", "日期"])
preds21 = {}
for sd in SEEDS:
    mm = lgb.train(lgb_params(sd),
        lgb.Dataset(tr_21[cols], label=tr_21["rank1"], feature_name=cols),
        num_boost_round=111)
    preds21[sd] = mm.predict(seg21[cols].fillna(0))
seg21["p42"] = preds21[42]
seg21["bag5"] = np.mean([preds21[s] for s in SEEDS], axis=0)
r1, _ = rankic_window(seg21, "p42", 20210701, 20211231)
r5, _ = rankic_window(seg21, "bag5", 20210701, 20211231)
print(f"单模型={r1:.4f}  5种子bag={r5:.4f}  差={r5-r1:+.4f}")

# ICIR feval（mean/std）与 5日滚动均值RankIC feval
val_dates = val["日期"].values
def icir_feval(preds_, dataset):
    tmp = pd.DataFrame({"y": dataset.get_label(), "p": preds_, "d": val_dates})
    ics = pd.Series([spearmanr(g["p"], g["y"])[0] for _, g in tmp.groupby("d") if len(g) >= 50]).dropna()
    return ("icir", ics.mean() / ics.std(), True)
def smooth_rankic_feval(preds_, dataset):
    tmp = pd.DataFrame({"y": dataset.get_label(), "p": preds_, "d": val_dates})
    ics = pd.Series([spearmanr(g["p"], g["y"])[0] for _, g in tmp.groupby("d") if len(g) >= 50]).dropna()
    return ("srankic", ics.rolling(5, min_periods=1).mean().iloc[-1], True)

ds_val = lgb.Dataset(val[cols], label=val["rank1"], feature_name=cols, free_raw_data=False)
for name, fe in [("ICIR", icir_feval), ("5日滚动RankIC", smooth_rankic_feval)]:
    mf = lgb.train(lgb_params(42),
        lgb.Dataset(tr1[cols], label=tr1["rank1"], feature_name=cols),
        num_boost_round=500, valid_sets=[ds_val], feval=fe,
        callbacks=[lgb.early_stopping(50, verbose=False)])
    seg22["alt"] = mf.predict(seg22[cols].fillna(0))
    ric, _ = rankic_window(seg22, "alt", 20220101, 20221231)
    print(f"{name}早停 best_iter={mf.best_iteration} 2022RankIC={ric:.4f} (基准0.0482)")
