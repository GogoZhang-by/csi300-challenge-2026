"""本地模拟平台评测：120日窗口 + submission 包推理，验证端到端一致性"""
import sys, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, "src")
from data_loader import DataLoader
from validate import RankICEvaluator

# 动态加载 submission 包（模拟平台环境）
sub_dir = Path("submission").resolve()
sys.path.insert(0, str(sub_dir))
spec = importlib.util.spec_from_file_location("sub_inference", sub_dir / "inference.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["sub_inference"] = mod
spec.loader.exec_module(mod)

model = mod.InferenceModel(str(sub_dir / "model"))
print("✅ 模型加载成功")

dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
f33 = dl.load_factor33()
bar = dl.load_barra()
labels = dl.build_labels(md)

# 全部交易日历（MarketData 中出现即交易日）
cal = sorted(md["日期"].unique())
test_dates = [d for d in cal if 20220101 <= d <= 20221231]
print(f"2022 交易日: {len(test_dates)}")

records = []
for d in test_dates:
    # 平台行为：截至 d 的最近 120 个交易日
    idx = cal.index(d)
    window = cal[max(0, idx - 119): idx + 1]
    data = {
        "MarketData": md.loc[md["日期"].isin(window)],
        "Factor33_winsor": f33.loc[f33["日期"].isin(window)],
        "Barra_factor": bar.loc[bar["日期"].isin(window)],
    }
    as_of = f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}"
    pred = model.predict(as_of, data)
    pred["_日期"] = d
    records.append(pred)

all_pred = pd.concat(records)
merged = all_pred.merge(labels, left_on=["_日期", "code"], right_on=["日期", "代码"], how="left")
merged = merged.drop(columns=["date"]).rename(columns={"_日期": "date"})

# 全池评估（含停牌，不剔除 —— 官方口径）
r_all = RankICEvaluator.evaluate(merged.dropna(subset=["target"]))
print(f"\n=== 端到端 120日窗口 2022 全池(含停牌) ===")
print(f"1dIC={r_all['mean_rankic']:.4f} IR={r_all['rankicir']:.3f} 正IC占比={r_all['positive_ratio']:.2%} 天数={r_all['n_days']}")

# 覆盖检查
cov = merged.groupby("date")["code"].nunique()
print(f"每日股票数: min={cov.min()} max={cov.max()}")
