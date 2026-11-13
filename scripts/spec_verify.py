"""关键实测：用官方平台测不同 holding_period 下的 RankIC，判断标签周期是否匹配
同时测 Barra 优化器下行业中性化的效果
"""
import sys
sys.path.insert(0, "submission")
import pandas as pd
import numpy as np
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))
DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"

print("=== 实验1: 不同 holding_period 下的调仓日 RankIC（当前 rank20 模型）===")
for hp in [1, 5, 10, 20]:
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=hp,   # 调仓周期=持有期，对齐
        holding_period=hp,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    print(f"H={hp:2d}: RankIC={m['mean_rankic']:.4f} ICIR={m['rankic_ir']:.3f} "
          f"正IC%={m['positive_rankic_ratio']:.1%} IR={m['information_ratio']:.2f} "
          f"换手={m['turnover']:.0f}%/年")

print("\n=== 实验2: 固定 rebalance=5，只改 holding_period（解耦）===")
for hp in [1, 5]:
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=5, holding_period=hp,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    print(f"rebal=5 H={hp}: RankIC={m['mean_rankic']:.4f} ICIR={m['rankic_ir']:.3f} IR={m['information_ratio']:.2f}")
