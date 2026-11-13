"""验证修正后模型：官方平台 H=1 口径（对齐公榜 RankIC）"""
import sys
sys.path.insert(0, "submission")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))
DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"

# 公榜口径推测：H=1（次日O2O），rebalance=5
for method in ["top_k", "barra"]:
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method=method, top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    print(f"{method}: RankIC@1={m['mean_rankic']:.4f} ICIR={m['rankic_ir']:.3f} "
          f"IR={m['information_ratio']:.2f} 换手={m['turnover']:.0f}%/年")
