"""探针包本地平台预测：提交后用实际公榜值比对，验证反解参数"""
import sys
sys.path.insert(0, "submission_probe")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
cost_med = CostConfig(commission_rate=0.0003, slippage=0.001,
                      fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))
cost_light = CostConfig(commission_rate=0.00025, slippage=0.0005,
                        fee_schedule=(FeeScheduleEntry("2022-01-01", 0.0005, 0.00001),))

print("=" * 70)
print("探针弱包（60特征+原始收益标签）本地官方平台回测 2022（train<=2021 诚实OOS）")
print("=" * 70)
for name, cost in [("无成本", CostConfig()), ("轻成本", cost_light), ("中成本", cost_med)]:
    for method in ["top_k", "barra"]:
        engine = BacktestEngine(
            data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
            submission_dir="submission_probe", lookback=120,
            rebalance_interval=5, holding_period=1,
            optimizer_config=OptimizerConfig(method=method, top_k=50),
            cost_config=cost, friendly_output=False,
        )
        m = engine.run()
        print(f"{name:5s} {method:6s} | RankIC={m['mean_rankic']:+.4f} "
              f"IR={m['information_ratio']:+.2f} 换手={m['turnover']:.0f}%/年 "
              f"Score={0.7*m['mean_rankic']+0.3*m['information_ratio']:+.3f}")
