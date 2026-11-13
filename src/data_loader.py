"""
统一数据加载器 — 沪深300竞赛公开数据 (2016-2022)
文件组织: data/<table>/<year>/<month>/<YYYYMM>.parquet
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir)

    # ── Parquet 表 ──────────────────────────────────────────────────────────
    def _load_parquet_table(self, table: str, end_date: Optional[int] = None) -> pd.DataFrame:
        """读取某张表全部 Parquet，按日期排序；可选 end_date 截断。"""
        paths = sorted(self.root.joinpath(table).rglob("*.parquet"))
        frames = []
        for p in paths:
            df = pd.read_parquet(p)
            if end_date is not None:
                df = df.loc[df["日期"].le(end_date)].copy()
            frames.append(df)
        if not frames:
            raise ValueError(f"No parquet found for table {table} under {self.root}")
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["日期", "代码"]).reset_index(drop=True)
        return out

    def load_market_data(self, end_date: Optional[int] = None) -> pd.DataFrame:
        return self._load_parquet_table("MarketData", end_date)

    def load_market_data_raw_open(self, end_date: Optional[int] = None) -> pd.DataFrame:
        return self._load_parquet_table("MarketDataRawOpen", end_date)

    def load_factor33(self, end_date: Optional[int] = None) -> pd.DataFrame:
        return self._load_parquet_table("Factor33_winsor", end_date)

    def load_barra(self, end_date: Optional[int] = None) -> pd.DataFrame:
        return self._load_parquet_table("Barra_factor", end_date)

    # ── CSV 参考表 ──────────────────────────────────────────────────────────
    def load_hs300_index(self) -> pd.DataFrame:
        frames = []
        for p in sorted(self.root.joinpath("HS300_index").rglob("hs300_index_*.csv")):
            frames.append(pd.read_csv(p, encoding="gb18030", dtype={"代码": "string"}))
        return pd.concat(frames, ignore_index=True)

    def load_hs300_industry(self) -> pd.DataFrame:
        frames = []
        for p in sorted(self.root.joinpath("HS300_industry").rglob("hs300_industry_*.csv")):
            frames.append(pd.read_csv(p, encoding="gb18030", dtype={"代码": "string"}))
        return pd.concat(frames, ignore_index=True)

    def load_hs300_weight(self) -> pd.DataFrame:
        frames = []
        for p in sorted(self.root.joinpath("HS300_weight").rglob("hs300_weight_*.csv")):
            frames.append(pd.read_csv(p, encoding="gb18030", dtype={"代码": "string"}))
        return pd.concat(frames, ignore_index=True)

    # ── 合并核心数据 ────────────────────────────────────────────────────────
    def load_daily_universe(self, date: int) -> list[str]:
        """返回某日的合法股票池（Barra_factor 中的代码）。"""
        barra = self.load_barra(end_date=date)
        codes = barra.loc[barra["日期"].eq(date), "代码"].unique().tolist()
        return codes

    def load_merged_features(self, end_date: Optional[int] = None) -> pd.DataFrame:
        """
        合并 MarketData + Factor33 + Barra 三张表。
        注意：合并键为 (日期, 代码)；MarketData/Factor33 是年度并集，Barra 是当日成分股。
        返回的 DataFrame 保留三张表全部列，后续应以 Barra 为准筛选股票池。
        """
        md = self.load_market_data(end_date)
        f33 = self.load_factor33(end_date)
        bar = self.load_barra(end_date)

        merged = bar.merge(
            md, on=["日期", "代码"], how="left", suffixes=("", "_md"), indicator=False
        ).merge(
            f33, on=["日期", "代码"], how="left", suffixes=("", "_f33"), indicator=False
        )
        # 名称列去重
        if "名称_md" in merged.columns:
            merged = merged.drop(columns=["名称_md"])
        if "名称_f33" in merged.columns:
            merged = merged.drop(columns=["名称_f33"])
        return merged

    # ── 标签构造 ────────────────────────────────────────────────────────────
    def build_labels(
        self, market: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算 O2O 标签:
          target    = O(t+2)/O(t+1) - 1     (官方评测口径, H=1)
          target_5  = O(t+6)/O(t+1) - 1     (5日平滑标签)
          target_20 = O(t+21)/O(t+1) - 1    (20日慢标签, 低换手)
        market 必须包含 ['日期','代码','open']
        返回 (日期, 代码, target, target_5, target_20) 其中日期 = 信号日 t
        """
        df = market[["日期", "代码", "open"]].copy()
        df = df.sort_values(["代码", "日期"]).reset_index(drop=True)
        g = df.groupby("代码")["open"]
        df["o1"] = g.shift(-1)
        df["o2"] = g.shift(-2)
        df["o6"] = g.shift(-6)
        df["o21"] = g.shift(-21)
        df["target"] = df["o2"] / df["o1"] - 1.0
        df["target_5"] = df["o6"] / df["o1"] - 1.0
        df["target_20"] = df["o21"] / df["o1"] - 1.0
        return df[["日期", "代码", "target", "target_5", "target_20"]]
