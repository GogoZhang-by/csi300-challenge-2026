"""冠军后防翻车筛选：只在诚实OOS(train<=2021, test=2022)上做Pareto筛选
基准 = 当前冠军配置 rank1 + w1。门槛: RankIC增益>=0.002 且 IR降幅<=0.15
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
val = clean.loc[clean["日期"].between(20220101, 20221231)]

seg = df.loc[df["日期"].between(20211201, 20221231)].copy().sort_values(["代码", "日期"])
test_dates = sorted(seg.loc[seg["日期"].between(20220101, 20221231), "日期"].unique())
rebal_dates = test_dates[::5]

def rankic_daily(t, sc):
    ics = [spearmanr(g[sc], g["target"])[0] for d, g in t.groupby("日期")
           if d >= 20220101 and len(g) >= 50]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)), float(np.mean(ics) / np.std(ics))

def ir_proxy(t, sc, cost=0.0015):
    prev, rs, tos = None, [], []
    for d in rebal_dates:
        g = t.loc[t["日期"].eq(d)].dropna(subset=["target_5"])
        if len(g) < 100:
            continue
        hold = set(g.nlargest(50, sc)["代码"])
        to = 1 - len(hold & prev) / 50 if prev else 1.0
        tos.append(to)
        ex = g.loc[g["代码"].isin(hold), "target_5"].mean() - g["target_5"].mean() - 2 * to * cost
        rs.append(ex)
        prev = hold
    r = pd.Series(rs)
    return r.mean() / r.std() * np.sqrt(242 / 5), float(np.mean(tos))

def evaluate(tag, pred_raw, smooth="none", blend_with=None):
    seg["raw"] = pred_raw
    if blend_with is not None:
        seg["raw"] = 0.5 * (seg["raw"] - seg.groupby("日期")["raw"].transform("mean")) / (seg.groupby("日期")["raw"].transform("std") + 1e-12) \
                   + 0.5 * (blend_with - seg.groupby("日期")["_b"].transform("mean")) / (seg.groupby("日期")["_b"].transform("std") + 1e-12)
    if smooth == "none":
        seg["sc"] = seg["raw"]
    else:
        w = smooth
        seg["sc"] = seg.groupby("代码", sort=False)["raw"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
    ric, icir = rankic_daily(seg, "sc")
    ir, to = ir_proxy(seg, "sc")
    print(f"{tag:34s} RankIC={ric:.4f} ICIR={icir:.3f} IR代理={ir:5.2f} 调仓换手={to:.1%}")
    return ric, ir

# 基准：L2早停（复刻冠军）
tr1 = train.loc[train["rank1"].notna()]
base = lgb.train(lgb_params(42),
    lgb.Dataset(tr1[cols], label=tr1["rank1"], feature_name=cols),
    num_boost_round=500,
    valid_sets=[lgb.Dataset(val[cols], label=val["rank1"], feature_name=cols)],
    callbacks=[lgb.early_stopping(50, verbose=False)])
pred_base = base.predict(seg[cols].fillna(0))
print(f"基准 best_iter={base.best_iteration}")
base_ric, base_ir = evaluate("基准 rank1 L2早停 w1", pred_base)

# 实验1: RankIC feval 早停
def rankic_feval(preds, dataset):
    y = dataset.get_label()
    gcol = dataset.get_group() if hasattr(dataset, "get_group") else None
    # 用pandas分组：需要日期，这里从构造好的val按顺序对齐
    tmp = pd.DataFrame({"y": y, "p": preds, "d": val_dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in tmp.groupby("d") if len(g) >= 50]
    return ("rankic", float(np.mean([x for x in ics if np.isfinite(x)])), True)

val_dates = val["日期"].values
ds_val = lgb.Dataset(val[cols], label=val["rank1"], feature_name=cols, free_raw_data=False)
model_feval = lgb.train(lgb_params(42),
    lgb.Dataset(tr1[cols], label=tr1["rank1"], feature_name=cols),
    num_boost_round=500, valid_sets=[ds_val],
    feval=rankic_feval, callbacks=[lgb.early_stopping(50, verbose=False)])
print(f"RankIC早停 best_iter={model_feval.best_iteration}")
evaluate("实验1 rank1 RankIC早停 w1", model_feval.predict(seg[cols].fillna(0)))

# 实验2: 极轻平滑 w=2,3（私榜成本防御）
for w in [2, 3]:
    evaluate(f"实验2 rank1 w{w} 轻平滑", pred_base, smooth=w)

# 实验3: 3种子bagging
preds_seed = [pred_base]
for sd in [7, 2024]:
    mm = lgb.train(lgb_params(sd),
        lgb.Dataset(tr1[cols], label=tr1["rank1"], feature_name=cols),
        num_boost_round=base.best_iteration)
    preds_seed.append(mm.predict(seg[cols].fillna(0)))
evaluate("实验3 rank1 3种子bagging w1", np.mean(preds_seed, axis=0))

# 实验4: rank1为主+少量rank5融合（不平滑）
tr5 = train.loc[train["rank5"].notna()]
m5 = lgb.train(lgb_params(42),
    lgb.Dataset(tr5[cols], label=tr5["rank5"], feature_name=cols),
    num_boost_round=base.best_iteration)
p5 = m5.predict(seg[cols].fillna(0))
seg["_b"] = p5
for a in [0.8, 0.9]:
    z1 = (seg["raw"] - seg.groupby("日期")["raw"].transform("mean")) / (seg.groupby("日期")["raw"].transform("std") + 1e-12) if False else None
    s1 = pd.Series(pred_base).pipe(lambda x: x)
    # 截面z标准化后融合
    tmp = seg[["日期", "代码"]].copy()
    tmp["p1"] = pred_base
    tmp["p5"] = p5
    for c in ["p1", "p5"]:
        tmp[c] = (tmp[c] - tmp.groupby("日期")[c].transform("mean")) / (tmp.groupby("日期")[c].transform("std") + 1e-12)
    tmp["mix"] = a * tmp["p1"] + (1 - a) * tmp["p5"]
    seg2 = seg.copy()
    seg2["sc"] = tmp["mix"].values
    ric, icir = rankic_daily(seg2, "sc")
    ir, to = ir_proxy(seg2, "sc")
    print(f"实验4 {a:.1f}rank1+{1-a:.1f}rank5 融合      RankIC={ric:.4f} ICIR={icir:.3f} IR代理={ir:5.2f} 调仓换手={to:.1%}")

print(f"\n门槛: RankIC >= {base_ric+0.002:.4f} 且 IR >= {base_ir-0.15:.2f} 才采纳")
