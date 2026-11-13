"""
训练管道 — 消融实验验证后的稳健方案：单一 LightGBM Regression。
- 标签: target_5 (5日平滑O2O, 降噪), 评估用 target (1日O2O, 官方口径)
- 切分: train ≤2020 / val 2021 (early stop) / test 2022 (诚实OOS)
- 最终模型: 全量 ≤2022 重训, 轮数取 val early stop 结果
产出: submission/ (inference.py + features.py + cleaning.py + model/)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import DataLoader
from features import build_features, get_feature_columns
from validate import RankICEvaluator


def metrics(df: pd.DataFrame, score_col: str, label_col: str = "target") -> dict:
    return RankICEvaluator.evaluate(
        df.rename(columns={"日期": "date"}), label_col=label_col, score_col=score_col
    )


def build_dataset(data_dir: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    dl = DataLoader(data_dir)
    md = dl.load_market_data()
    data = {
        "MarketData": md,
        "Factor33_winsor": dl.load_factor33(),
        "Barra_factor": dl.load_barra(),
    }
    feats = build_features(data)
    labels = dl.build_labels(md)
    df = feats.merge(labels, on=["日期", "代码"], how="left", validate="one_to_one")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def lgb_params(seed: int) -> dict:
    # 消融实验中表现稳健的参数 (与 baseline 一致)
    return {
        "objective": "regression", "metric": "l2", "verbosity": -1,
        "learning_rate": 0.04, "num_leaves": 15, "max_depth": 6,
        "min_data_in_leaf": 100, "feature_fraction": 0.85,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "seed": seed, "num_threads": 8, "deterministic": True,
        "force_col_wise": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data_temp/Data"))
    ap.add_argument("--cache", type=Path, default=Path("result/feature_cache.parquet"))
    ap.add_argument("--submission-dir", type=Path, default=Path("submission"))
    ap.add_argument("--label-col", default="target_rank")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = get_feature_columns()
    label_col = args.label_col
    print(f"特征数: {len(cols)}, 训练标签: {label_col}")
    df = build_dataset(args.data_dir, args.cache)

    # 截面 rank 标签（实验验证大幅提升 RankIC：原始收益0.034→rank标签0.044+）
    # target_H_rank = 每日 target_H 的截面百分位 - 0.5 ∈ [-0.5, 0.5]
    # H=5 与平台 holding_period=5 口径对齐（实测 RankIC 峰值在 H=5）
    rank_map = {"target_rank": "target", "target_5_rank": "target_5", "target_20_rank": "target_20"}
    if label_col in rank_map:
        raw = rank_map[label_col]
        df[label_col] = df.groupby("日期")[raw].rank(pct=True) - 0.5

    # 清洗（诊断验证口径）
    mask = (
        df["target"].notna() & df[label_col].notna()
        & df["is_suspend"].eq(0) & df["close"].gt(0)
        & df["target"].abs().le(0.25) & df[label_col].abs().le(0.60)
        & df["list_days"].ge(60)
    )
    clean = df.loc[mask].copy()
    print(f"样本: 总{len(df)} -> 清洗后{len(clean)}")

    train = clean.loc[clean["日期"].le(20211231)]
    val = clean.loc[clean["日期"].between(20220101, 20221231)]
    print(f"train {len(train)} | val {len(val)}")

    # 训练 (early stop on 2022 val，因为目标公榜/私榜更接近 2022 regime)
    m = lgb.train(
        lgb_params(args.seed),
        lgb.Dataset(train[cols], label=train[label_col], feature_name=cols),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val[cols], label=val[label_col], feature_name=cols)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iter = m.best_iteration
    val = val.assign(score=m.predict(val[cols], num_iteration=best_iter))

    print(f"\nbest_iter={best_iter}")
    print(f"val 2022 (early stop): 1dIC={metrics(val,'score')['mean_rankic']:.4f} "
          f"5dIC={metrics(val,'score','target_5')['mean_rankic']:.4f} "
          f"IR1d={metrics(val,'score')['rankicir']:.3f}")

    # 特征重要性 top15
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\nTop15 特征:")
    print(imp.head(15).to_string())

    # ── 全量重训 (≤2022) 固化 ──────────────────────────────────────────────
    final = clean.loc[clean["日期"].le(20221231)]
    m_final = lgb.train(
        lgb_params(args.seed),
        lgb.Dataset(final[cols], label=final[label_col], feature_name=cols),
        num_boost_round=best_iter,
    )

    model_dir = args.submission_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    m_final.save_model(str(model_dir / "lgb_reg.txt"))
    meta = {
        "feature_columns": cols,
        "label_col": label_col,
        "seed": args.seed,
        "best_iter": best_iter,
        "score_avg_window": 20,
        "train_range": [20160104, 20221231],
        "oos_2022": metrics(val, "score"),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 拷贝推理包源码
    root = Path(__file__).resolve().parent.parent
    shutil.copyfile(root / "src" / "features.py", args.submission_dir / "features.py")
    shutil.copyfile(root / "submission_src" / "inference.py", args.submission_dir / "inference.py")
    print(f"\n✅ 提交包已生成: {args.submission_dir}")


if __name__ == "__main__":
    main()
