"""对齐竞赛口径：1日RankIC + 真实费用 + Top-K优化器（最可能的公榜配置）"""
import sys
sys.path.insert(0, "submission")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig

for holding_period in [1, 5]:
    for method, top_k in [("top_k", 50), ("barra", 50)]:
        engine = BacktestEngine(
            data_dir=r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data",
            start_date="2022-01-01",
            end_date="2022-12-31",
            submission_dir="submission",
            lookback=120,
            rebalance_interval=5,
            holding_period=holding_period,
            optimizer_config=OptimizerConfig(method=method, top_k=top_k),
            cost_config=CostConfig(
                commission_rate=0.0003,       # 佣金 万3
                slippage=0.001,               # 滑点 0.1%
                fee_schedule=(__import__("skd_backtest").FeeScheduleEntry(
                    effective_date="2022-01-01",
                    stamp_tax_rate=0.001,      # 印花税 千1（卖出）
                    transfer_fee_rate=0.00002, # 过户费 万0.2
                ),),
            ),
            friendly_output=False,
        )
        m = engine.run()
        print(f"\n[holding={holding_period}d, {method} top{top_k}, 含成本]")
        print(f"  RankIC={m['mean_rankic']:.4f}  RankICIR={m['rankic_ir']:.3f}  正IC%={m['positive_rankic_ratio']:.1%}")
        print(f"  年化超额={m['annualized_excess_return']:.2%}  跟踪误差={m['tracking_error']:.2%}  IR={m['information_ratio']:.3f}")
        print(f"  年化换手={m['turnover']:.0f}%  费用合计={m['transaction_cost']:.4f}")
        print(f"  公式分(0.7*RankIC+0.3*IR)={0.7*m['mean_rankic']+0.3*m['information_ratio']:.3f}")
