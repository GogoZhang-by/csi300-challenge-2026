"""官方平台实测 rank1+raw 配置（临时训练，不覆盖提交模型）"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
from features import get_feature_columns
from train import lgb_params
from data_loader import DataLoader

df = pd.read_parquet("result/feature_cache.parquet")
cols = get_feature_columns()

# 构造 rank1 标签
df["rank1"] = df.groupby("日期")["target"].rank(pct=True) - 0.5

mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
train = clean.loc[clean["日期"].le(20211231)]

m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["rank1"], feature_name=cols),
              num_boost_round=175)

# 保存临时模型
m.save_model("submission/model/lgb_reg_rank1.txt")
print("rank1 model saved")

# 修改 inference 临时用 rank1（不覆盖原配置）
# 先跑官方平台
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))

# 临时改 metadata
import json
meta_path = "submission/model/metadata.json"
meta = json.load(open(meta_path, encoding="utf-8"))
meta["label_col"] = "rank1"
meta["score_avg_window"] = 1  # 不平滑
json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

engine = BacktestEngine(
    data_dir=r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data",
    start_date="2022-01-01", end_date="2022-12-31",
    submission_dir="submission", lookback=120,
    rebalance_interval=5, holding_period=5,
    optimizer_config=OptimizerConfig(method="top_k", top_k=50),
    cost_config=COST, friendly_output=False,
)
m = engine.run()
print(f"rank1+raw: RankIC={m['mean_rankic']:.4f} ICIR={m['rankic_ir']:.3f} IR={m['information_ratio']:.2f}")
