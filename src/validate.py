"""
时序验证框架 — walk-forward 切分 + RankIC 指标计算。
严格保证 train < val < test 时间顺序，无泄露。
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class Fold:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardSplitter:
    """
    Walk-forward 时序切分。
    默认策略: expanding window
    例: 2016-2019 train / 2020 val / 2021 test
        2016-2020 train / 2021 val / 2022 test
    """

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def split(self, dates: np.ndarray) -> List[Fold]:
        """dates: 升序排列的 YYYYMMDD int 数组"""
        dates = np.sort(np.unique(dates))
        n = len(dates)
        # 按年数大致均分，最后一年作为 test
        years = np.array([d // 10000 for d in dates])
        unique_years = np.unique(years)
        if len(unique_years) < 3:
            raise ValueError("至少需要3年数据才能做 train/val/test 切分")

        folds = []
        # 倒数第 n_splits+1 年开始作为 val/test
        for i in range(self.n_splits):
            # 每年后移一年
            val_year = unique_years[-(self.n_splits + 1 - i)]
            test_year = unique_years[-(self.n_splits - i)]
            train_end_year = val_year - 1

            train_dates = dates[years <= train_end_year]
            val_dates = dates[years == val_year]
            test_dates = dates[years == test_year]

            if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
                continue

            folds.append(Fold(
                train_start=int(train_dates.min()),
                train_end=int(train_dates.max()),
                val_start=int(val_dates.min()),
                val_end=int(val_dates.max()),
                test_start=int(test_dates.min()),
                test_end=int(test_dates.max()),
            ))
        return folds


class RankICEvaluator:
    """计算 Mean RankIC / RankICIR / 正IC占比"""

    @staticmethod
    def daily_rankic(scores: pd.Series, labels: pd.Series) -> float:
        if len(scores) < 10:
            return np.nan
        ic, _ = spearmanr(scores, labels)
        return float(ic)

    @staticmethod
    def evaluate(pred_df: pd.DataFrame, label_col: str = "target", score_col: str = "score") -> dict:
        """
        pred_df 必须包含: date(YYYYMMDD int), target, score
        """
        daily = pred_df.groupby("date").apply(
            lambda g: RankICEvaluator.daily_rankic(g[score_col], g[label_col])
        ).dropna()
        return {
            "mean_rankic": float(daily.mean()),
            "std_rankic": float(daily.std()),
            "rankicir": float(daily.mean() / daily.std()) if daily.std() > 0 else np.nan,
            "positive_ratio": float((daily > 0).mean()),
            "n_days": int(len(daily)),
        }
