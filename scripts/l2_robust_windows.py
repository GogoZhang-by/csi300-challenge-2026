"""l2=1.0 关键稳健性：2021H2（train<=2020, ES val=2021H1, 测2021H2）+ 2022H2（train<=202203, val Q2, 测H2）"""
import sys, json, importlib
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import pandas as pd
from pathlib import Path

train_mod = importlib.import_module("train")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2021-01-01", 0.001, 0.00002),))

df = pd.read_parquet("result/oos_test_cache.parquet")
cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]

def run_platform(start, end, fee_year="2021"):
    def iso(d):
        s = str(d)
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    cost = CostConfig(commission_rate=0.0003, slippage=0.001,
                      fee_schedule=(FeeScheduleEntry(f"{fee_year}-01-01", 0.001, 0.00002),))
    Path("submission_test/model/metadata.json").write_text(
        json.dumps({"feature_columns": cols, "score_avg_window": 1,
                    "ir_mode": "raw", "ir_lambda": 0.0}, ensure_ascii=False), encoding="utf-8")
    engine = BacktestEngine(
        data_dir=DATA, start_date=iso(start), end_date=iso(end),
        submission_dir="submission_test", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=cost, friendly_output=False,
    )
    m = engine.run()
    return m["mean_rankic"], m["information_ratio"]

def window_test(name, tr_hi, val_lo, val_hi, test_lo, test_hi, fee_year):
    trd = clean.loc[clean["日期"].le(tr_hi)]
    vald = clean.loc[clean["日期"].between(val_lo, val_hi)]
    out = {}
    for l2 in [5.0, 1.0]:
        params = train_mod._lgb_params(42); params["lambda_l2"] = l2
        m = lgb.train(params, lgb.Dataset(trd[cols], label=trd["target_rank"], feature_name=cols),
                      num_boost_round=500,
                      valid_sets=[lgb.Dataset(vald[cols], label=vald["target_rank"], feature_name=cols)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        bi = m.best_iteration
        # 最终模型：train+val 合并，测 test
        finald = clean.loc[clean["日期"].le(val_hi)]
        mf = lgb.train(params, lgb.Dataset(finald[cols], label=finald["target_rank"], feature_name=cols),
                       num_boost_round=bi)
        mf.save_model("submission_test/model/lgb_reg.txt")
        out[l2] = (*run_platform(test_lo, test_hi, fee_year), bi)
    d_ric = out[1.0][0] - out[5.0][0]
    d_ir = out[1.0][1] - out[5.0][1]
    print(f"{name}: l2=5 RankIC={out[5.0][0]:.4f} IR={out[5.0][1]:5.2f} | "
          f"l2=1 RankIC={out[1.0][0]:.4f} IR={out[1.0][1]:5.2f} | "
          f"ΔRankIC={d_ric:+.4f} ΔIR={d_ir:+.2f} ΔScore={0.7*d_ric+0.3*d_ir:+.3f}")
    return d_ric, d_ir

print("=== l2=1.0 vs 5.0 多窗口稳健性（官方平台，早停）===")
d1 = window_test("2021H2", 20201231, 20210101, 20210630, 20210701, 20211231, "2021")
d2 = window_test("2022H1", 20210630, 20210701, 20211231, 20220101, 20220630, "2022")
d3 = window_test("2022H2", 20211231, 20220101, 20220630, 20220701, 20221231, "2022")
wins = sum(1 for d in (d1, d2, d3) if 0.7*d[0]+0.3*d[1] > 0)
print(f"\n3窗口中 l2=1.0 胜出 {wins}/3 → {'稳健，可用于最终提交' if wins >= 2 else '不稳健，放弃'}")
