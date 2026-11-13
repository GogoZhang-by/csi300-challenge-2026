"""图2验证：在 barra LP 优化器（私榜口径）下，测试分数预中性化是否真的提升 IR
train<=2021 诚实OOS，2022测试。约束：单股/行业/Barra风格/换手
"""
import sys, json, glob
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
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))
BARRA = [f"barra_{n}" for n in ("市值","贝塔","动量","残差波动","非线性市值","账面市值比","流动性","盈利","成长","杠杆")]

# ── 数据与OOS模型 ──
df = pd.read_parquet("result/oos_test_cache.parquet")
cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
tr = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]
m = lgb.train(train_mod._lgb_params(42),
    lgb.Dataset(tr[cols], label=tr["target_rank"], feature_name=cols), num_boost_round=111)
seg = df.loc[df["日期"].between(20220101, 20221231)].copy().sort_values(["日期","代码"])
seg["raw"] = m.predict(seg[cols].fillna(0))

# ── 行业映射（GBK CSV）──
ind_files = sorted(glob.glob(f"{DATA}/HS300_industry/2022/*.csv"))
ind = pd.concat([pd.read_csv(f, encoding="gbk") for f in ind_files], ignore_index=True)
ind["日期"] = ind["日期"].astype(int)
seg = seg.merge(ind[["日期","代码","行业代码"]], on=["日期","代码"], how="left")
seg["行业代码"] = seg["行业代码"].fillna("NA")

# ── 分数变换 ──
def csrank(s): return s.groupby("日期")["raw"].rank(pct=True)

def industry_resid(s):
    """每日对行业虚拟变量回归取残差"""
    out = pd.Series(index=s.index, dtype=float)
    for d, idx in s.groupby("日期").groups.items():
        g = s.loc[idx]
        D = pd.get_dummies(g["行业代码"], drop_first=True).astype(float).values
        X = np.column_stack([np.ones(len(g)), D])
        b_, *_ = np.linalg.lstsq(X, g["raw"].values, rcond=None)
        out.loc[idx] = g["raw"].values - X @ b_
    return out.groupby(s["日期"]).rank(pct=True)

def barra_resid(s, factors):
    out = pd.Series(index=s.index, dtype=float)
    for d, idx in s.groupby("日期").groups.items():
        g = s.loc[idx]
        X = np.column_stack([np.ones(len(g)), g[factors].fillna(0).values])
        b_, *_ = np.linalg.lstsq(X, g["raw"].values, rcond=None)
        out.loc[idx] = g["raw"].values - X @ b_
    return out.groupby(s["日期"]).rank(pct=True)

def vol_resid(s):
    z = s.groupby("日期")["raw"].rank(pct=True) - 0.5*s.groupby("日期")["vol_20"].rank(pct=True)
    return z.groupby(s["日期"]).rank(pct=True)

print("计算分数变换...")
seg["s_raw"] = csrank(seg)
seg["s_ind"] = industry_resid(seg)
seg["s_barra"] = barra_resid(seg, BARRA)
seg["s_indbarra"] = barra_resid(seg.assign(raw=seg["s_ind"]), BARRA)
seg["s_vol"] = vol_resid(seg)

def run(score_col, oc, tag):
    s = seg[["日期","代码",score_col]].rename(columns={"日期":"d","代码":"c",score_col:"score"})
    s.to_parquet("submission_test/model/_pred.parquet", index=False)
    Path("submission_test/model/metadata.json").write_text(json.dumps(
        {"feature_columns":cols,"score_avg_window":1}), encoding="utf-8")
    eng = BacktestEngine(data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission_test", lookback=120, rebalance_interval=5, holding_period=1,
        optimizer_config=oc, cost_config=COST, friendly_output=False)
    r = eng.run()
    print(f"{tag:30s} RankIC={r['mean_rankic']:.4f} IR={r['information_ratio']:6.2f} "
          f"TE={r['tracking_error']:.3f} 超额={r['annualized_excess_return']:+.3f} "
          f"换手={r['turnover']:.0f}% 拒单={r['failed_orders']} Score={0.7*r['mean_rankic']+0.3*r['information_ratio']:.3f}")

# ── 三档约束（私榜公式的 λ/η 未知，扫松紧）──
tight = OptimizerConfig(method="barra",
    single_name_weight_limit=0.03,
    industry_exposure_limit=0.03, barra_style_exposure_limit=0.15)
medium = OptimizerConfig(method="barra",
    single_name_weight_limit=0.05, industry_exposure_limit=0.05,
    barra_style_exposure_limit=0.30)
loose = OptimizerConfig(method="barra", single_name_weight_limit=0.05)

for name, oc in [("紧约束(行业3%/风格0.15/换手25%)", tight),
                 ("中约束(行业5%/风格0.30/换手40%)", medium),
                 ("松约束(仅单股5%)", loose)]:
    print(f"\n===== {name} =====")
    for col, lab in [("s_raw","原始分数"), ("s_ind","行业中性"),
                     ("s_barra","Barra中性"), ("s_indbarra","行业+Barra中性"), ("s_vol","波动中性")]:
        try:
            run(col, oc, lab)
        except Exception as e:
            print(f"{lab:30s} ERROR {type(e).__name__}: {str(e)[:80]}")
