"""D11-12 鲁棒性测试：多种子 + 特征加噪"""
import sys
sys.path.insert(0, "src")
import lightgbm as lgb
import pandas as pd
import numpy as np
from features import get_feature_columns
from validate import RankICEvaluator
from train import lgb_params, build_dataset
from pathlib import Path

cols = get_feature_columns()
df = build_dataset(Path("data_temp/Data"), Path("result/feature_cache.parquet"))

mask = (df["target"].notna() & df["target_5"].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df["target_5"].abs().le(0.60)
        & df["list_days"].ge(60))
clean = df.loc[mask].copy()
train = clean.loc[clean["日期"].le(20211231)]
val = clean.loc[clean["日期"].between(20220101, 20221231)]

def ev(df_, col, label="target"):
    return RankICEvaluator.evaluate(df_.rename(columns={"日期": "date"}), label_col=label, score_col=col)

# 1. 多种子
print("=== 多种子稳定性 ===")
ics = []
for seed in [42, 7, 2024]:
    m = lgb.train(lgb_params(seed), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
                  num_boost_round=500,
                  valid_sets=[lgb.Dataset(val[cols], label=val["target_5"], feature_name=cols)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    v = val.assign(s=m.predict(val[cols], num_iteration=m.best_iteration))
    r = ev(v, "s")
    ics.append(r["mean_rankic"])
    print(f"seed={seed}: iter={m.best_iteration} 1dIC={r['mean_rankic']:.4f} IR={r['rankicir']:.3f}")
print(f"种子间 1dIC 标准差: {np.std(ics):.4f} (均值 {np.mean(ics):.4f})")

# 2. 特征加噪（模拟真实数据小毛病）
print("\n=== 加噪鲁棒性（对 val 特征加 5% 高斯噪声）===")
m = lgb.train(lgb_params(42), lgb.Dataset(train[cols], label=train["target_5"], feature_name=cols),
              num_boost_round=500,
              valid_sets=[lgb.Dataset(val[cols], label=val["target_5"], feature_name=cols)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
best_iter = m.best_iteration
base = val.assign(s=m.predict(val[cols], num_iteration=best_iter))
r0 = ev(base, "s")
print(f"无噪声: 1dIC={r0['mean_rankic']:.4f} IR={r0['rankicir']:.3f}")

rng = np.random.default_rng(0)
for noise in [0.01, 0.05, 0.10]:
    noisy = val.copy()
    num_cols = [c for c in cols if noisy[c].dtype != object]
    stds = noisy[num_cols].std().replace(0, 1)
    noisy[num_cols] = noisy[num_cols] + pd.DataFrame(
        rng.normal(0, noise, noisy[num_cols].shape) * stds.to_numpy(),
        index=noisy.index, columns=num_cols)
    vn = noisy.assign(s=m.predict(noisy[cols], num_iteration=best_iter))
    rn = ev(vn, "s")
    print(f"噪声{noise:.0%}: 1dIC={rn['mean_rankic']:.4f} (衰减 {(r0['mean_rankic']-rn['mean_rankic'])/max(abs(r0['mean_rankic']),1e-9):.1%}) IR={rn['rankicir']:.3f}")

# 3. 缺失值鲁棒性（随机屏蔽 10% 特征值）
print("\n=== 缺失鲁棒性（随机置 NaN 10%）===")
miss = val.copy()
mask_nan = rng.random(miss[cols].shape) < 0.10
miss[cols] = miss[cols].mask(mask_nan)
vm = miss.assign(s=m.predict(miss[cols].fillna(0), num_iteration=best_iter))
rm = ev(vm, "s")
print(f"10%缺失: 1dIC={rm['mean_rankic']:.4f} (衰减 {(r0['mean_rankic']-rm['mean_rankic'])/max(abs(r0['mean_rankic']),1e-9):.1%}) IR={rm['rankicir']:.3f}")
