"""强正则+中性化 双窗口验证（2022 + 2021H2独立窗）"""
import sys, json
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

def _load(n, p):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
train_mod = _load("sub_train", "submission/train.py")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
df = pd.read_parquet("result/oos_test_cache.parquet")
cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
BARRA = [f"barra_{n}" for n in ("市值","贝塔","动量","残差波动","非线性市值","账面市值比","流动性","盈利","成长","杠杆")]

def fit(tr, val, seed=42, lr=0.02, ff=0.6, leaves=31, es=True):
    p = train_mod._lgb_params(seed)
    p.update({"learning_rate": lr, "feature_fraction": ff, "num_leaves": leaves})
    kw = {"num_boost_round": 500}
    if es and val is not None:
        kw["valid_sets"]=[lgb.Dataset(val[cols], label=val["target_rank"], feature_name=cols)]
        kw["callbacks"]=[lgb.early_stopping(50, verbose=False)]
    return lgb.train(p, lgb.Dataset(tr[cols], label=tr["target_rank"], feature_name=cols), **kw)

def run(seg, score_fn, cost_year, start, end):
    s = seg.copy(); s["score"] = score_fn(s); s = s.dropna(subset=["score"])
    s.rename(columns={"日期":"d","代码":"c"})[["d","c","score"]].to_parquet("submission_test/model/_pred.parquet", index=False)
    Path("submission_test/model/metadata.json").write_text(json.dumps({"feature_columns":cols,"score_avg_window":1}), encoding="utf-8")
    eng = BacktestEngine(data_dir=DATA, start_date=start, end_date=end,
        submission_dir="submission_test", lookback=120, rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=CostConfig(commission_rate=0.0003, slippage=0.001,
            fee_schedule=(FeeScheduleEntry(f"{cost_year}-01-01",0.001,0.00002),)),
        friendly_output=False)
    m = eng.run(); return m["mean_rankic"], m["information_ratio"], m["tracking_error"]

def csrank(s): return s.groupby("日期")["raw"].rank(pct=True)
def vol_neu(s, lam): return csrank(s) - lam*s.groupby("日期")["vol_20"].rank(pct=True)
def beta_neu(s, lam): return csrank(s) - lam*s.groupby("日期")["barra_贝塔"].rank(pct=True)

def window(name, tr_hi, val_lo, val_hi, te_lo, te_hi, cost_year):
    trd = clean.loc[clean["日期"].le(tr_hi)]
    vald = clean.loc[clean["日期"].between(val_lo, val_hi)] if val_lo else None
    seg = df.loc[df["日期"].between(te_lo, te_hi)].copy()
    out = {}
    for tag, kw in [("基准", dict(lr=0.04, ff=0.85, leaves=15)),
                    ("强正则", dict(lr=0.02, ff=0.6, leaves=31))]:
        m = fit(trd, vald, **kw)
        seg["raw"] = m.predict(seg[cols].fillna(0))
        for vn, fn in [("raw", csrank), ("vol.5", lambda s: vol_neu(s,0.5)),
                       ("beta.5", lambda s: beta_neu(s,0.5))]:
            out[f"{tag}+{vn}"] = run(seg, fn, cost_year, f"{str(te_lo)[:4]}-{str(te_lo)[4:6]}-{str(te_lo)[6:]}", f"{str(te_hi)[:4]}-{str(te_hi)[4:6]}-{str(te_hi)[6:]}")
    print(f"\n=== {name} ===")
    for k,(ric,ir,te) in out.items():
        print(f"  {k:16s} RankIC={ric:.4f} IR={ir:5.2f} TE={te:.3f} Score={0.7*ric+0.3*ir:.3f}")
    return out

window("2022全年", 20211231, 20220101, 20221231, 20220101, 20221231, "2022")
window("2021H2", 20201231, 20210101, 20210630, 20210701, 20211231, "2021")
