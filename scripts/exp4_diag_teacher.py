"""建议4 快速诊断：截面自注意力教师（单窗 2022，快速版）
- 快速 rank(pct)：groupby.rank 一次算全部特征
- 梯度累积（每 16 天一步）+ 15 epoch，快速验证教师分数是否有信号
- 同时测：教师分数 RankIC + 「基础前60 + 教师尾部重排」RankIC
"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm

from dual_window_validate import _df, _clean, _BASE_COLS, daily_rankic, ir_backtest, lgb_params

D_MODEL = 32
N_HEADS = 4
TOP_KEEP = 60
ACC_STEPS = 16
EPOCHS = 15
torch.manual_seed(42)


class Teacher(nn.Module):
    def __init__(self, d_in=71, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.wq = nn.Linear(d_in, d_model)
        self.wk = nn.Linear(d_in, d_model)
        self.wv = nn.Linear(d_in, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        n = x.shape[0]
        q = self.wq(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        k = self.wk(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        v = self.wv(x).view(n, self.n_heads, self.d_head).transpose(0, 1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.d_head ** 0.5), dim=-1)
        out = (attn @ v).transpose(0, 1).reshape(n, -1)
        return self.head(out).squeeze(-1)


def rank_pct(df):
    return df.groupby("日期")[_BASE_COLS].rank(pct=True).fillna(0.5)


def train_teacher(train_end: int):
    tr = _clean.loc[_clean["日期"].le(train_end)]
    feats = rank_pct(tr)
    model = Teacher()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # 按日分组，收集每日张量
    days = []
    for d, g in tr.groupby("日期"):
        X = torch.tensor(feats.loc[g.index].to_numpy(dtype=np.float32))
        y = torch.tensor(g["target_rank"].to_numpy(dtype=np.float32))
        days.append((X, y))
    opt.zero_grad()
    for ep in range(EPOCHS):
        total, cnt = 0.0, 0
        for i, (X, y) in enumerate(days):
            loss = nn.functional.mse_loss(model(X), y)
            (loss / ACC_STEPS).backward()
            total += loss.item() * len(y)
            cnt += len(y)
            if (i + 1) % ACC_STEPS == 0:
                opt.step()
                opt.zero_grad()
        if len(days) % ACC_STEPS != 0:
            opt.step()
            opt.zero_grad()
        print(f"    ep{ep:02d} mse={total/cnt:.5f}", flush=True)
    return model


def predict_teacher(model, seg):
    feats = rank_pct(seg)
    rows = []
    with torch.no_grad():
        for d, g in seg.groupby("日期"):
            X = torch.tensor(feats.loc[g.index].to_numpy(dtype=np.float32))
            rows.append(pd.DataFrame({"日期": g["日期"].to_numpy(), "代码": g["代码"].to_numpy(),
                                      "t_score": model(X).numpy()}))
    return pd.concat(rows, ignore_index=True)


def tail_rerank(seg, base_col, enh_col):
    final = seg[base_col].to_numpy().astype(float).copy()
    for d, g in seg.groupby("日期"):
        idx = g.index
        base = g[base_col].to_numpy()
        enh = g[enh_col].to_numpy()
        order = np.argsort(-base, kind="stable")
        top, tail = order[:TOP_KEEP], order[TOP_KEEP:]
        if len(tail) == 0:
            continue
        min_top = base[top].min()
        r = pd.Series(enh[tail]).rank(ascending=True, method="first").to_numpy()
        r = (r - 1) / len(tail)
        final[idx[tail]] = min_top - 1.0 + r
    return final


if __name__ == "__main__":
    print("=" * 70)
    print(f"建议4快速诊断：截面自注意力教师 (d={D_MODEL},{N_HEADS}头, {EPOCHS}ep) 主窗2022")
    print("=" * 70)
    tr_base = _clean.loc[_clean["日期"].le(20211231)]
    m_base = lightgbm.train(lgb_params(42),
                            lightgbm.Dataset(tr_base[_BASE_COLS], label=tr_base["target_rank"], feature_name=_BASE_COLS),
                            num_boost_round=175)
    teacher = train_teacher(20211231)

    seg = _df.loc[_df["日期"].between(20220101, 20221231)].copy()
    seg["base"] = m_base.predict(seg[_BASE_COLS].fillna(0.0))
    tpred = predict_teacher(teacher, seg)
    seg = seg.merge(tpred, on=["日期", "代码"], how="left")
    seg["t_score"] = seg["t_score"].fillna(0.0)
    seg["final"] = tail_rerank(seg, "base", "t_score")

    for col, name in [("base", "纯基础LGB"), ("t_score", "教师分数"), ("final", "基础前60+教师尾部")]:
        ric = daily_rankic(seg, col)
        print(f"  [{name}] RankIC={ric['mean_rankic']:.4f} ICIR={ric['rankicir']:.3f} 正IC%={ric['positive_ratio']:.1%}")
