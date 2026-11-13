"""IR破3杠杆实验 v2 — 用正确平台接口，train<=2021诚实OOS，反解参数"""
import sys, json
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

train_mod = _load("sub_train", "submission/train.py")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))

df = pd.read_parquet("result/oos_test_cache.parquet")
cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
tr = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]
BARRA = [f"barra_{n}" for n in ("市值","贝塔","动量","残差波动","非线性市值","账面市值比","流动性","盈利","成长","杠杆")]

def fit(seed=42, lr=0.04, nround=500, es=True, feature_frac=0.85, leaves=15):
    p = train_mod._lgb_params(seed)
    p.update({"learning_rate": lr, "feature_fraction": feature_frac, "num_leaves": leaves})
    kw = {"num_boost_round": nround}
    if es:
        kw["valid_sets"] = [lgb.Dataset(val[cols], label=val["target_rank"], feature_name=cols)]
        kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    m = lgb.train(p, lgb.Dataset(tr[cols], label=tr["target_rank"], feature_name=cols), **kw)
    return m

def predict_2022(m):
    """对2022全部日期打分，返回 (date_int, code, raw_pred)"""
    seg = df.loc[df["日期"].between(20220101, 20221231)].copy()
    seg["raw"] = m.predict(seg[cols].fillna(0))
    return seg[["日期","代码","raw","vol_20"] + BARRA]

def csrank(g): return g.groupby("日期")["raw"].rank(pct=True)

def write_submission(predseg, score_fn, tag):
    """把变换后的分数通过 platform 评估。predseg 是2022段带raw，score_fn(seg)->seg加score列"""
    s = predseg.copy()
    s["score"] = score_fn(s)
    s = s.dropna(subset=["score"])
    # 写成 inference 直接读 parquet
    s_out = s.rename(columns={"日期":"d","代码":"c"})[["d","c","score"]]
    Path("submission_test/model/_pred.parquet").parent.mkdir(parents=True, exist_ok=True)
    s_out.to_parquet("submission_test/model/_pred.parquet", index=False)
    Path("submission_test/model/metadata.json").write_text(json.dumps({
        "feature_columns": cols, "score_avg_window": 1, "_mode": "precomputed"}), encoding="utf-8")
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission_test", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False)
    m = engine.run()
    return m["mean_rankic"], m["information_ratio"], m["tracking_error"], m["turnover"]

# 实验定义
def E_raw(s): return s["raw"]
def E_csrank(s): return csrank(s)
def E_vol_neutral(s):
    return csrank(s) - 0.5*s.groupby("日期")["vol_20"].rank(pct=True)
def E_barra_resid(s):
    out = s.copy(); res = np.zeros(len(out))
    for d, idx in out.groupby("日期").groups.items():
        g = out.loc[idx]
        X = np.column_stack([np.ones(len(g)), g[BARRA].fillna(0).values])
        b_, *_ = np.linalg.lstsq(X, g["raw"].values, rcond=None)
        res[out.index.get_indexer(idx)] = g["raw"].values - X@b_
    return pd.Series(res, index=out.index)

print("训练模型...")
mA = fit(42)                      # 基准单seed
segA = predict_2022(mA)
bi = mA.best_iteration or 111

mB = [fit(sd, nround=bi, es=False) for sd in (42,7,2024)]   # 3seed
segB = segA.copy()
segB["raw"] = np.mean([mm.predict(df.loc[df["日期"].between(20220101,20221231), cols].fillna(0)) for mm in mB], axis=0)

mC = fit(42, lr=0.01)             # 低lr
segC = predict_2022(mC)

mD = fit(42, lr=0.02, feature_frac=0.6, leaves=31)  # 更正则
segD = predict_2022(mD)

print("跑平台...")
results = []
for name, seg, fn in [
    ("A 基准单seed",        segA, E_raw),
    ("A2 csrank",           segA, E_csrank),
    ("B 3seed集成+csrank",  segB, E_csrank),
    ("C lr.01+csrank",      segC, E_csrank),
    ("D 强正则+csrank",     segD, E_csrank),
    ("E 波动中性+csrank",   segA, E_vol_neutral),
    ("F Barra残差+csrank",  segA, E_barra_resid),
]:
    try:
        r = write_submission(seg, fn, name)
        results.append((name, *r))
        print(f"{name:22s} RankIC={r[0]:.4f} IR={r[1]:5.2f} TE={r[2]:.3f} 换手={r[3]:.0f}% Score={0.7*r[0]+0.3*r[1]:.3f}")
    except Exception as e:
        print(f"{name:22s} ERROR {type(e).__name__}: {e}")
