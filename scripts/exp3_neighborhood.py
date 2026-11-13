"""
建议3+1 实验：风格邻域成对排序器（增强分数）+ 尾部重排输出层
- 邻域：标准化 Barra 风格距离的 top-k 近邻（同日内）
- pair 样本：特征差 X_i - X_j，label = sign(target_i - target_j)，权重 = |Δtarget| 封顶
- 推理：股票 i 的增强分数 = 对邻域内 pairwise 得分(概率)的均值
- 输出层：基础模型 rank 前 60 名原序不动，其余按增强分数重排
"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors

from dual_window_validate import (
    _df, _clean, _BASE_COLS, daily_rankic, ir_backtest, lgb_params,
)

STYLE_COLS = [
    "barra_市值", "barra_贝塔", "barra_动量", "barra_残差波动", "barra_非线性市值",
    "barra_账面市值比", "barra_流动性", "barra_盈利", "barra_成长", "barra_杠杆",
]
K = 8          # 近邻数
CAP = 0.15     # 收益差封顶
TOP_KEEP = 60  # 尾部重排：保留头部前 N 名原序


def _standardize_style(df: pd.DataFrame) -> np.ndarray:
    """按日截面 z-score 标准化 Barra 风格，NaN→0"""
    S = df[STYLE_COLS].copy()
    for c in STYLE_COLS:
        g = df.groupby("日期")[c]
        S[c] = (S[c] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return S.fillna(0.0).to_numpy(dtype=np.float32)


def _build_pairs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X_diff, label, weight)，只包含 |Δtarget|>1e-6 的 pair"""
    X = df[_BASE_COLS].to_numpy(dtype=np.float32)
    tgt = df["target"].to_numpy(dtype=np.float32)
    S = _standardize_style(df)
    dates = df["日期"].to_numpy()

    xs, ys, ws = [], [], []
    for d in np.unique(dates):
        m = dates == d
        Xd = X[m]
        Sd = S[m]
        td = tgt[m]
        n = len(Xd)
        if n < K + 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(K + 1, n)).fit(Sd)
        ind = nn.kneighbors(Sd, return_distance=False)  # (n, k+1)
        neighbors = ind[:, 1:]  # 去掉自身
        kk = neighbors.shape[1]
        i_idx = np.repeat(np.arange(n), kk)
        j_idx = neighbors.reshape(-1)
        diff = Xd[i_idx] - Xd[j_idx]
        dt = td[i_idx] - td[j_idx]
        keep = np.abs(dt) > 1e-6
        xs.append(diff[keep])
        ys.append((dt[keep] > 0).astype(np.float32))
        ws.append(np.minimum(np.abs(dt[keep]), CAP))
    Xd = np.vstack(xs).astype(np.float32)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    return Xd, y, w


def _train_neighbor(train_end: int):
    tr = _clean.loc[_clean["日期"].le(train_end)]
    Xd, y, w = _build_pairs(tr)
    print(f"    pair样本: {len(y)} 正例占比={(y==1).mean():.3f}")
    p = lgb_params(42)
    p.update({"objective": "binary", "metric": "binary_logloss",
              "min_data_in_leaf": 200, "lambda_l2": 1.0})
    ds = lgb.Dataset(Xd, label=y, weight=w)
    return lgb.train(p, ds, num_boost_round=120)


def _predict_enhanced(model, seg: pd.DataFrame) -> pd.DataFrame:
    """返回 (日期, 代码, 增强分数=邻居胜率)"""
    X = seg[_BASE_COLS].to_numpy(dtype=np.float32)
    S = _standardize_style(seg)
    dates = seg["日期"].to_numpy()
    codes = seg["代码"].to_numpy()
    out_d, out_c, out_s = [], [], []
    for d in np.unique(dates):
        m = dates == d
        Xd = X[m]
        Sd = S[m]
        n = len(Xd)
        if n < K + 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(K + 1, n)).fit(Sd)
        ind = nn.kneighbors(Sd, return_distance=False)
        neighbors = ind[:, 1:]
        kk = neighbors.shape[1]
        i_idx = np.repeat(np.arange(n), kk)
        j_idx = neighbors.reshape(-1)
        diff = Xd[i_idx] - Xd[j_idx]
        p = model.predict(diff).reshape(n, kk).mean(axis=1)
        out_d.append(np.full(n, d, dtype=int))
        out_c.append(codes[m])
        out_s.append(p)
    return pd.DataFrame({
        "日期": np.concatenate(out_d),
        "代码": np.concatenate(out_c),
        "enh": np.concatenate(out_s),
    })


def _tail_rerank(seg: pd.DataFrame, base_col: str, enh_col: str) -> np.ndarray:
    """基础前 TOP_KEEP 名原序，其余按增强分数重排（输出分数量级与 base 对齐、尾部低于头部）"""
    final = seg[base_col].to_numpy().astype(float).copy()
    for d, g in seg.groupby("日期"):
        idx = g.index
        base = g[base_col].to_numpy()
        enh = g[enh_col].to_numpy()
        order = np.argsort(-base, kind="stable")  # 降序
        top = order[:TOP_KEEP]
        tail = order[TOP_KEEP:]
        if len(tail) == 0:
            continue
        min_top = base[top].min()
        # 尾部按增强分数重排：增强越高 → 尾部分数越高（仍严格低于头部）
        r = pd.Series(enh[tail]).rank(ascending=True, method="first").to_numpy()
        r = (r - 1) / len(tail)
        final[idx[tail]] = min_top - 1.0 + r  # 严格低于头部
    return final


def _run(train_end: int, test_start: int, test_end: int, tag: str):
    # 基础模型
    tr_base = _clean.loc[_clean["日期"].le(train_end)]
    m_base = lgb.train(
        lgb_params(42),
        lgb.Dataset(tr_base[_BASE_COLS], label=tr_base["target_rank"], feature_name=_BASE_COLS),
        num_boost_round=175,
    )
    # 邻域成对模型
    m_nb = _train_neighbor(train_end)

    seg = _df.loc[_df["日期"].between(test_start, test_end)].copy()
    seg["base"] = m_base.predict(seg[_BASE_COLS].fillna(0.0))
    enh = _predict_enhanced(m_nb, seg)
    seg = seg.merge(enh, on=["日期", "代码"], how="left")
    seg["enh"] = seg["enh"].fillna(0.5)
    seg["final"] = _tail_rerank(seg, "base", "enh")

    for col, name in [("base", "纯基础"), ("enh", "增强分数(邻域胜率)"), ("final", "邻域+尾部重排")]:
        ric = daily_rankic(seg, col)
        ir = ir_backtest(seg, col, f"{test_start//10000}-{test_start%10000//100:02d}-{test_start%100:02d}",
                         f"{test_end//10000}-{test_end%10000//100:02d}-{test_end%100:02d}")
        print(f"  [{tag} {name}] RankIC={ric['mean_rankic']:.4f} ICIR={ric['rankicir']:.3f} "
              f"正IC%={ric['positive_ratio']:.1%} | IR={ir['information_ratio']:.3f} TE={ir['tracking_error']:.3f}")


if __name__ == "__main__":
    print("=" * 70)
    print(f"建议3+1：风格邻域成对排序器 (K={K}, CAP={CAP}) + 尾部重排 (保留前{TOP_KEEP})")
    print("=" * 70)
    _run(20211231, 20220101, 20221231, "主窗 2022")
    _run(20210630, 20210701, 20211231, "副窗 2021H2")
