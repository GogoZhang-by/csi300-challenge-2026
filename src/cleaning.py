"""
数据清洗 — 区分训练期 fit 与推理期 transform，所有统计量固化到 model/。
关键处理：停牌价格静止检测、缺失值填充、极端值缩尾、新上市过滤。
"""
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd


class Cleaner:
    """
    两阶段清洗器：
    1) 训练时 fit_transform 从训练数据学习填充统计量并应用清洗
    2) 推理时 load + transform 用已学习统计量清洗新数据
    """

    def __init__(self):
        self.stats: dict = {}          # 填充统计量 {col: median}
        self.numeric_cols: list = []   # 参与缩尾的数值列
        self.is_fitted = False

    # ── 训练阶段 ────────────────────────────────────────────────────────────
    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[list] = None,
        winsorize: bool = True,
        winsor_q: float = 0.005,
        min_list_days: int = 60,
    ) -> pd.DataFrame:
        """
        训练时一次性完成 fit + transform。
        - numeric_cols: 需要填充+缩尾的数值列；None 则自动推断
        - winsorize: 是否做对称缩尾
        - winsor_q: 两端缩尾分位数（默认 0.5%，对应约 3σ）
        - min_list_days: 上市天数不足则标记为无效（不删除，留 is_valid 标记）
        """
        df = df.copy()

        # 1. 停牌日价格静止标记
        df["_susp_price_frozen"] = (
            df["is_suspend"].eq(1)
            & df["volume"].eq(0)
        ).astype(int)

        # 2. 自动推断数值列（排除标识列和已标记列）
        if numeric_cols is None:
            exclude = {"日期", "代码", "名称", "is_suspend", "is_st", "is_new", "list_days", "_susp_price_frozen"}
            numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        self.numeric_cols = numeric_cols

        # 3. 学习填充统计量：仅对非停牌静止样本计算中位数
        for col in numeric_cols:
            valid = df.loc[df["_susp_price_frozen"].eq(0), col]
            self.stats[col] = float(valid.median())

        # 4. 缺失值填充
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(self.stats[col])

        # 5. 缩尾（winsorize）：训练时学习上下界
        self.winsor_bounds = {}
        if winsorize:
            for col in numeric_cols:
                lo = df[col].quantile(winsor_q)
                hi = df[col].quantile(1 - winsor_q)
                self.winsor_bounds[col] = (float(lo), float(hi))
                df[col] = df[col].clip(lo, hi)

        # 6. 新上市 / 复牌过滤标记
        df["_is_valid"] = 1
        if "list_days" in df.columns:
            df.loc[df["list_days"].lt(min_list_days), "_is_valid"] = 0
        # 停牌静止日训练时标记为无效样本（不影响预测时输出score，仅用于训练剔除）
        df.loc[df["_susp_price_frozen"].eq(1), "_is_valid"] = 0

        self.is_fitted = True
        return df

    # ── 保存/加载 ───────────────────────────────────────────────────────────
    def save(self, model_dir: Path | str) -> None:
        path = Path(model_dir) / "cleaner_stats.json"
        payload = {
            "stats": self.stats,
            "numeric_cols": self.numeric_cols,
            "winsor_bounds": self.winsor_bounds,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path | str) -> "Cleaner":
        path = Path(model_dir) / "cleaner_stats.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        inst = cls()
        inst.stats = {k: float(v) for k, v in payload["stats"].items()}
        inst.numeric_cols = payload["numeric_cols"]
        inst.winsor_bounds = {k: tuple(v) for k, v in payload.get("winsor_bounds", {}).items()}
        inst.is_fitted = True
        return inst

    # ── 推理阶段 ────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """推理时清洗，不重新 fit，用训练期统计量。"""
        if not self.is_fitted:
            raise RuntimeError("Cleaner has not been fitted. Call fit_transform first or load().")
        df = df.copy()

        # 停牌标记（推理时也要标记，用于特征工程决策）
        df["_susp_price_frozen"] = (
            df["is_suspend"].eq(1)
            & df["volume"].eq(0)
        ).astype(int)

        # 填充 + 缩尾
        for col in self.numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(self.stats.get(col, 0.0))
            if col in self.winsor_bounds:
                lo, hi = self.winsor_bounds[col]
                df[col] = df[col].clip(lo, hi)

        # 推理时 _is_valid 仅标记新上市，不停牌剔除（必须输出全部300只score）
        df["_is_valid"] = 1
        if "list_days" in df.columns:
            df.loc[df["list_days"].lt(60), "_is_valid"] = 0
        return df
