"""用主办方回测平台跑我们的 submission，拿到官方口径 RankIC / IR"""
import sys
sys.path.insert(0, "submission")
from pathlib import Path
from skd_backtest import BacktestEngine, OptimizerConfig

engine = BacktestEngine(
    data_dir=r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data",
    start_date="2022-01-01",
    end_date="2022-12-31",
    submission_dir="submission",           # 官方标准提交目录
    lookback=120,                          # 赛题约束：≤120 交易日
    rebalance_interval=5,                  # 平台默认：5日调仓
    holding_period=5,                      # RankIC 持有期
    optimizer_config=OptimizerConfig(method="barra", top_k=50),
    output_dir="result/official_backtest",
)
metrics = engine.run()
print("\n=== 官方回测原始指标 ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")
