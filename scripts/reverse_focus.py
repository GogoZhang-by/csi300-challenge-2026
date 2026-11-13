"""聚焦反解：只测最可能的6组参数"""
import sys
sys.path.insert(0, "submission")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"

cost_none = CostConfig()
cost_light = CostConfig(commission_rate=0.00025, slippage=0.0005,
                        fee_schedule=(FeeScheduleEntry("2022-01-01", 0.0005, 0.00001),))
cost_med = CostConfig(commission_rate=0.0003, slippage=0.001,
                      fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))

configs = [
    ("reb5 top50 none",   5, "top_k", 50,  cost_none),
    ("reb5 top50 light",  5, "top_k", 50,  cost_light),
    ("reb5 top50 medium", 5, "top_k", 50,  cost_med),
    ("reb5 top100 med",   5, "top_k", 100, cost_med),
    ("reb5 top30 med",    5, "top_k", 30,  cost_med),
    ("reb5 barra50 med",  5, "barra", 50,  cost_med),
    ("reb1 top50 med",    1, "top_k", 50,  cost_med),
]

for name, reb, method, k, cost in configs:
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=reb, holding_period=1,
        optimizer_config=OptimizerConfig(method=method, top_k=k),
        cost_config=cost, friendly_output=False,
    )
    m = engine.run()
    print(f"{name:22s} | RankIC={m['mean_rankic']:.4f} IR={m['information_ratio']:5.2f} "
          f"换手={m['turnover']:.0f}%/年 成本={m['transaction_cost']:.4f}")

print("\n公榜目标: RankIC=0.0635 IR=2.6374")
