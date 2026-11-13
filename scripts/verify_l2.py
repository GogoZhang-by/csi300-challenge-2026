"""用官方平台验证 lambda_l2=1.0（2022, 反解参数）"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
from features import get_feature_columns
from train import lgb_params, build_dataset
from pathlib import Path

df = build_dataset(Path("data_temp/Data"), Path("result/feature_cache.parquet"))
cols = get_feature_columns()
df["rank1"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]

for l2, name in [(0.0, "baseline"), (1.0, "l2=1.0")]:
    params = {**lgb_params(42), "lambda_l2": l2}
    m = lgb.train(params, lgb.Dataset(train[cols], label=train["rank1"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(val[cols], label=val["rank1"], feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    # 存临时模型
    m.save_model(f"submission/model/lgb_reg_{name.replace('=','_')}.txt")
    print(f"{name}: best_iter={m.best_iteration}")

# 改 metadata 跑两个版本
import json
meta = json.load(open("submission/model/metadata.json", encoding="utf-8"))

from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry
DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))

import shutil
for name in ["baseline", "l2=1.0"]:
    fname = name.replace("=", "_")
    shutil.copy(f"submission/model/lgb_reg_{fname}.txt", "submission/model/lgb_reg.txt")
    meta["label_col"] = "target_rank"
    meta["score_avg_window"] = 1
    json.dump(meta, open("submission/model/metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    print(f"[{name}] RankIC={m['mean_rankic']:.4f} IR={m['information_ratio']:.2f} "
          f"换手={m['turnover']:.0f}%")
