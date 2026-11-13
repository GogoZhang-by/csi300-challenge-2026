"""验证新模型（target_5_rank + dmean）在官方平台的表现"""
import sys
sys.path.insert(0, "submission")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))
DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"

for method in ["top_k", "barra"]:
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission", lookback=120,
        rebalance_interval=5, holding_period=5,
        optimizer_config=OptimizerConfig(method=method, top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    ric = m['mean_rankic']
    icir = m['rankic_ir']
    ir = m['information_ratio']
    to = m['turnover']
    print(f"{method}: RankIC={ric:.4f} ICIR={icir:.3f} IR={ir:.2f} 换手={to:.0f}%")
