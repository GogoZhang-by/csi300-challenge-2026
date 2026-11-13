"""
D1 EDA — 验证关键假设
1. Barra_factor 每日恰好300只（当日成分股）
2. Factor33/MarketData 是年度并集（>300只）
3. 停牌日价格静止（volume=0, 价格不变）
4. 标签 O(t+2)/O(t+1)-1 构造正确
"""
import sys
sys.path.insert(0, "src")
from data_loader import DataLoader
import pandas as pd
import numpy as np

dl = DataLoader("data_temp/Data")

print("=" * 60)
print("假设1: Barra_factor = 当日真实成分股 (300只/日)")
bar = dl.load_barra()
daily_cnt = bar.groupby("日期")["代码"].nunique()
print(f"  每日股票数: min={daily_cnt.min()}, max={daily_cnt.max()}, mean={daily_cnt.mean():.1f}")
print(f"  总天数: {len(daily_cnt)}")

print("\n" + "=" * 60)
print("假设2: Factor33/MarketData 是年度并集 (每日>300)")
md = dl.load_market_data()
f33 = dl.load_factor33()
md_cnt = md.groupby("日期")["代码"].nunique()
f33_cnt = f33.groupby("日期")["代码"].nunique()
print(f"  MarketData 每日: min={md_cnt.min()}, max={md_cnt.max()}, mean={md_cnt.mean():.1f}")
print(f"  Factor33 每日: min={f33_cnt.min()}, max={f33_cnt.max()}, mean={f33_cnt.mean():.1f}")

print("\n" + "=" * 60)
print("假设3: 停牌日价格静止 (volume=0 且 open=high=low=close)")
# 用 2016-01 熔断月验证
m2016 = md.loc[md["日期"].between(20160101, 20160131)]
sus = m2016.loc[m2016["is_suspend"].eq(1)]
same_price = (sus["open"] == sus["high"]) & (sus["high"] == sus["low"]) & (sus["low"] == sus["close"])
print(f"  2016-01 停牌样本: {len(sus)}, 价格全同比例: {same_price.mean():.2%}, volume=0比例: {(sus['volume']==0).mean():.2%}")

print("\n" + "=" * 60)
print("假设4: 标签 O(t+2)/O(t+1)-1 构造")
# 手工验证一只股票
m_one = md.loc[md["代码"].eq("SH600000")].sort_values("日期").head(10).copy()
print("  浦发银行前10日 open:")
print(m_one[["日期", "open"]].to_string(index=False))
# t=第一天, target 应 = open[2]/open[1] - 1
labels = dl.build_labels(md.loc[md["代码"].eq("SH600000")])
first = labels.iloc[0]
manual = m_one["open"].iloc[2] / m_one["open"].iloc[1] - 1
print(f"  信号日 {first['日期']}: 自动标签={first['target']:.6f}, 手工计算={manual:.6f}, 一致={np.isclose(first['target'], manual)}")

print("\n" + "=" * 60)
print("假设5: 合并后 Barra 为左表，应无缺失的 open/close")
merged = dl.load_merged_features()
# 检查2022年某日
day2022 = merged.loc[merged["日期"].eq(20220104)]
print(f"  2022-01-04 Barra 股票数: {len(day2022)}, open 缺失: {day2022['open'].isna().sum()}, close 缺失: {day2022['close'].isna().sum()}")
print(f"  列示例: {day2022.columns.tolist()[:10]}")

print("\n✅ D1 EDA 完成")
