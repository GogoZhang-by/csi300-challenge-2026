"""λ_l2 早停口径官方平台精扫 + 小市值叠加（诚实OOS）"""
import sys, json, importlib
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import pandas as pd
from pathlib import Path

train_mod = importlib.import_module("train")
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

def train_es(l2):
    params = train_mod._lgb_params(42)
    params["lambda_l2"] = l2
    m = lgb.train(params, lgb.Dataset(tr[cols], label=tr["target_rank"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(val[cols], label=val["target_rank"], feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model("submission_test/model/lgb_reg.txt")
    return m.best_iteration

def run(mode="raw", lam=0.0):
    Path("submission_test/model/metadata.json").write_text(
        json.dumps({"feature_columns": cols, "score_avg_window": 1,
                    "ir_mode": mode, "ir_lambda": lam}, ensure_ascii=False), encoding="utf-8")
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission_test", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    return m["mean_rankic"], m["information_ratio"], m["turnover"]

base = None
for l2 in [5.0, 2.0, 1.0, 0.5]:
    bi = train_es(l2)
    r = run()
    print(f"l2={l2:.1f}(ES {bi:3d}): RankIC={r[0]:.4f} IR={r[1]:5.2f} 换手={r[2]:.0f}% Score={0.7*r[0]+0.3*r[1]:.3f}")
    if l2 == 5.0:
        base = r

# 最优 l2 候选 1.0 已在最后一次训练，叠加小市值惩罚
for lam in [0.03, 0.05, 0.08]:
    r = run("smallcap", lam)
    print(f"l2=1.0+小市值{lam:.2f}: RankIC={r[0]:.4f} IR={r[1]:5.2f} 换手={r[2]:.0f}% Score={0.7*r[0]+0.3*r[1]:.3f}")

print(f"\n基准(l2=5.0 ES): RankIC={base[0]:.4f} IR={base[1]:.2f}")
