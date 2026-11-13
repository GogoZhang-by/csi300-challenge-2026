"""
D2 — 本地复现 baseline RankIC（2022年）
用官方冻结模型在 2022 年公开数据上模拟评测，验证本地 RankIC 计算与官方口径一致。
"""
import sys
sys.path.insert(0, "baseline_temp/baseline_submission 2")
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data_loader import DataLoader

# 1. 加载 baseline 模型
from inference import InferenceModel
model = InferenceModel("baseline_temp/baseline_submission 2/model")
print("✅ Baseline 模型加载成功")

# 2. 加载 2022 年数据
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
f33 = dl.load_factor33()
bar = dl.load_barra()
labels = dl.build_labels(md)

# 2022 年交易日
test_dates = sorted([d for d in bar["日期"].unique() if 20220101 <= d <= 20221231])
print(f"2022 年交易日数: {len(test_dates)}")

# 3. 每日推理 + RankIC
daily_rankic = []
for d in test_dates:
    data = {
        "MarketData": md.loc[md["日期"].eq(d)],
        "Factor33_winsor": f33.loc[f33["日期"].eq(d)],
        "Barra_factor": bar.loc[bar["日期"].eq(d)],
    }
    pred = model.predict(f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}", data)
    # 合并标签（pred.date 是字符串，labels.日期 是 int）
    pred["_日期"] = pred["date"].str.replace("-", "").astype(int)
    merged = pred.merge(labels, left_on=["_日期", "code"], right_on=["日期", "代码"], how="left")
    merged = merged.dropna(subset=["target"])
    if len(merged) < 10:
        continue
    ic, _ = spearmanr(merged["score"], merged["target"])
    daily_rankic.append({"date": d, "rankic": ic, "n_stocks": len(merged)})

ric = pd.DataFrame(daily_rankic)
print("\n" + "=" * 50)
print("Baseline 2022 年本地复现结果")
print(f"  Mean RankIC:  {ric['rankic'].mean():.4f}")
print(f"  RankIC Std:   {ric['rankic'].std():.4f}")
print(f"  RankICIR:     {ric['rankic'].mean() / ric['rankic'].std():.4f}")
print(f"  正IC占比:     {(ric['rankic'] > 0).mean():.2%}")
print(f"  覆盖天数:     {len(ric)}")
print("=" * 50)

ric.to_csv("result/baseline_2022_rankic.csv", index=False)
print("已保存 result/baseline_2022_rankic.csv")
