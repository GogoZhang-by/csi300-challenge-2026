"""IR组合层实验：诚实OOS模型(train<=2021) × score变换，官方平台反解参数"""
import sys, json, importlib
sys.path.insert(0, "submission_test")
import lightgbm as lgb
import pandas as pd
from pathlib import Path

train_mod = importlib.import_module("train")
from skd_backtest import BacktestEngine, OptimizerConfig, CostConfig, FeeScheduleEntry

DATA = r"c:\Users\jingw\Desktop\coding\DT_quant_competition\data_temp\Data"
COST = CostConfig(commission_rate=0.0003, slippage=0.001,
                  fee_schedule=(FeeScheduleEntry("2022-01-01", 0.001, 0.00002),))

# ── 构建数据并训练诚实OOS模型（train<=2021）──
cache = Path("result/oos_test_cache.parquet")
if cache.exists():
    df = pd.read_parquet(cache)
else:
    md = train_mod._load_table(Path(DATA), "MarketData")
    f33 = train_mod._load_table(Path(DATA), "Factor33_winsor")
    bar = train_mod._load_table(Path(DATA), "Barra_factor")
    data = {"MarketData": md, "Factor33_winsor": f33, "Barra_factor": bar}
    feats = train_mod._build_features(data)
    labels = train_mod._build_labels(md)
    df = feats.merge(labels, on=["日期", "代码"], how="left", validate="one_to_one")
    df["target_rank"] = df.groupby("日期")["target"].rank(pct=True) - 0.5
    df.to_parquet(cache, index=False)

cols = train_mod._get_feature_columns()
mask = (df["target"].notna() & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["list_days"].ge(60))
clean = df.loc[mask]
tr = clean.loc[clean["日期"].le(20211231)]

def train_oos(l2):
    params = train_mod._lgb_params(42)
    params["lambda_l2"] = l2
    m = lgb.train(params, lgb.Dataset(tr[cols], label=tr["target_rank"], feature_name=cols),
                  num_boost_round=111)  # 冠军 best_iter
    m.save_model("submission_test/model/lgb_reg.txt")

def run(mode, lam):
    meta = {"feature_columns": cols, "score_avg_window": 1,
            "ir_mode": mode, "ir_lambda": lam}
    Path("submission_test/model/metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    engine = BacktestEngine(
        data_dir=DATA, start_date="2022-01-01", end_date="2022-12-31",
        submission_dir="submission_test", lookback=120,
        rebalance_interval=5, holding_period=1,
        optimizer_config=OptimizerConfig(method="top_k", top_k=50),
        cost_config=COST, friendly_output=False,
    )
    m = engine.run()
    return m["mean_rankic"], m["information_ratio"], m["turnover"]

train_oos(5.0)
print("l2=5.0 (冠军正则)")
base = run("raw", 0.0)
print(f"{'raw基准':16s} RankIC={base[0]:.4f} IR={base[1]:5.2f} 换手={base[2]:.0f}% Score={0.7*base[0]+0.3*base[1]:.3f}")
results = [("raw", 0.0, *base)]

for mode, lam, tag in [
    ("vol", 0.05, "波动惩罚λ.05"), ("vol", 0.10, "波动惩罚λ.10"),
    ("smallcap", 0.05, "小市值惩罚.05"),
    ("limitu", 0.05, "涨停惩罚.05"), ("limitu", 0.10, "涨停惩罚.10"),
    ("blend5", 0.2, "5日混合.2"), ("blend5", 0.3, "5日混合.3"), ("blend5", 0.5, "5日混合.5"),
    ("combo", 0.05, "组合.05"),
]:
    r = run(mode, lam)
    results.append((mode, lam, *r))
    print(f"{tag:16s} RankIC={r[0]:.4f} IR={r[1]:5.2f} 换手={r[2]:.0f}% Score={0.7*r[0]+0.3*r[1]:.3f}")

print("\n=== Pareto: 相对 raw 基准 ===")
b0 = base
for mode, lam, ric, ir, to in results:
    print(f"{mode:9s} λ={lam:.2f} ΔRankIC={ric-b0[0]:+.4f} ΔIR={ir-b0[1]:+.2f} "
          f"ΔScore={0.7*(ric-b0[0])+0.3*(ir-b0[1]):+.3f}")
