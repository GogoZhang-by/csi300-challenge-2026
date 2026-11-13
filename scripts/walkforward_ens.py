"""私榜泛化杠杆：walk-forward 多训练截止日集成 vs 单模型；双优化器验证"""
import sys, json
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

def _load(n,p):
    spec = importlib.util.spec_from_file_location(n,p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
train_mod = _load("sub_train","submission/train.py")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry
DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01",0.001,0.00002),))
df = pd.read_parquet("result/oos_test_cache.parquet")
cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
seg = df.loc[df["日期"].between(20220101,20221231)].copy().sort_values(["日期","代码"])

def train_cutoff(hi, nround=111):
    trd = clean.loc[clean["日期"].le(hi)]
    return lgb.train(train_mod._lgb_params(42),
        lgb.Dataset(trd[cols], label=trd["target_rank"], feature_name=cols), num_boost_round=nround)

print("训练3个截止日模型: <=2019 / <=2020 / <=2021")
models = {20191231: None, 20201231: None, 20211231: None}
for hi in models: models[hi] = train_cutoff(hi)

preds = {hi: mm.predict(seg[cols].fillna(0)) for hi, mm in models.items()}
seg["s_single"] = pd.Series(preds[20211231], index=seg.index).groupby(seg["日期"]).rank(pct=True)
ens3 = np.mean(list(preds.values()), axis=0)
seg["s_ens3"] = pd.Series(ens3, index=seg.index).groupby(seg["日期"]).rank(pct=True)
ens2 = 0.5*preds[20201231] + 0.5*preds[20211231]
seg["s_ens2"] = pd.Series(ens2, index=seg.index).groupby(seg["日期"]).rank(pct=True)

def run(score_col, oc, tag):
    seg[["日期","代码",score_col]].rename(columns={"日期":"d","代码":"c",score_col:"score"}).to_parquet(
        "submission_test/model/_pred.parquet", index=False)
    Path("submission_test/model/metadata.json").write_text(json.dumps({"feature_columns":cols,"score_avg_window":1}),encoding="utf-8")
    eng = BacktestEngine(data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission_test", lookback=120, rebalance_interval=5, holding_period=1,
        optimizer_config=oc, cost_config=COST, friendly_output=False)
    r = eng.run()
    print(f"{tag:26s} RankIC={r['mean_rankic']:.4f} IR={r['information_ratio']:5.2f} "
          f"TE={r['tracking_error']:.3f} 超额={r['annualized_excess_return']:+.3f} Score={0.7*r['mean_rankic']+0.3*r['information_ratio']:.3f}")

topk = OptimizerConfig(method="top_k", top_k=50)
barra = OptimizerConfig(method="barra", single_name_weight_limit=0.05)
for oname, oc in [("top_k", topk), ("barra松", barra)]:
    print(f"\n===== {oname} =====")
    for c, t in [("s_single","单模型<=2021"), ("s_ens2","2窗集成(20,21)"), ("s_ens3","3窗集成(19,20,21)")]:
        run(c, oc, t)
