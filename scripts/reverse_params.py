"""反解平台评测参数：扫 holding/rebalance/optimizer/top_k/cost，找最接近公榜 0.0635/2.6374 的组合"""
import sys, itertools
sys.path.insert(0, "submission")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
TARGET_RIC = 0.0635
TARGET_IR = 2.6374

costs = {
    "none": CostConfig(),
    "light": CostConfig(commission_rate=0.00025, slippage=0.0005,
                        fee_schedule=(FeeScheduleEntry("2022-01-01", 0.0005, 0.00001),)),
    "medium": CostConfig(commission_rate=0.0003, slippage=0.001,
                         fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),)),
    "heavy": CostConfig(commission_rate=0.0005, slippage=0.002,
                        fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),)),
}

results = []
for hp in [1, 5]:
    for reb in [1, 5]:
        for method in ["top_k", "barra"]:
            for k in [30, 50, 100]:
                for cname, cost in costs.items():
                    try:
                        engine = BacktestEngine(
                            data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
                            submission_dir="submission", lookback=120,
                            rebalance_interval=reb, holding_period=hp,
                            optimizer_config=OptimizerConfig(method=method, top_k=k),
                            cost_config=cost, friendly_output=False,
                        )
                        m = engine.run()
                        ric, ir, to = m["mean_rankic"], m["information_ratio"], m["turnover"]
                        # 与公榜的距离（标准化）
                        d = ((ric - TARGET_RIC) / 0.01) ** 2 + ((ir - TARGET_IR) / 0.3) ** 2
                        results.append((d, hp, reb, method, k, cname, ric, ir, to))
                        print(f"H={hp} reb={reb} {method:6s} k={k:3d} cost={cname:6s} | "
                              f"RankIC={ric:.4f} IR={ir:5.2f} 换手={to:.0f}%  dist={d:.2f}")
                    except Exception as e:
                        print(f"H={hp} reb={reb} {method} k={k} cost={cname}: ERROR {e}")

print("\n=== 最接近公榜的 Top 10 ===")
for d, hp, reb, method, k, cname, ric, ir, to in sorted(results)[:10]:
    print(f"H={hp} reb={reb} {method:6s} k={k:3d} cost={cname:6s} | "
          f"RankIC={ric:.4f} IR={ir:5.2f} 换手={to:.0f}%  dist={d:.2f}")
