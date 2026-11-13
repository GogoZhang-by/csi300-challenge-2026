<p align="center"><strong>⚠️ 本仓库内容仅供学习研究用 ⚠️</strong></p>
<p align="center"><strong>🚫 禁止以任何方式应用于实盘 🚫</strong></p>

# 沪深300截面选股 —— 冠军方案复刻与改进实验

本项目针对「沪深300指数增强/截面选股」评测口径（次日 open-to-open 收益的截面 RankIC）展开两部分工作：

1. **稳健基线**（`src/` + `submission/`）：71 维特征的单一 LightGBM 截面排序模型。
2. **冠军 FNR 方案复刻**（`champion/`）：完整复刻因子邻域排序（Factor Neighborhood Ranker）底座，并做工程层小改进。

所有结论以**本地时序外样本（OOS）**为准，不以任何公开榜单次为准。

## 目录结构

```
├── src/                   # 基线开发代码（数据加载/特征/验证/训练）
├── submission/            # 基线提交包（自包含推理入口 + 冻结权重）
├── champion/              # 冠军 FNR 方案复刻（vendored，含 Windows 兼容修复）
│   ├── inference.py       # 平台推理入口：InferenceModel
│   ├── train.py           # 完整离线训练入口
│   ├── requirements*.txt  # 推理 / 训练依赖
│   ├── ATTRIBUTION.md     # 原方案说明（归属声明）
│   └── fnr/               # inference / features / models / training / configs
├── scripts/               # 实验与双窗口 OOS 验证脚手架
├── notebooks/             # 探索性分析
└── tests/                 # smoke test
```

> 比赛原始数据（`data_temp/`）、回测结果（`result/`）与中间产物均不入库，请按赛事授权自行获取数据。

## 基线方案

71 维特征（49 baseline 因子 + 12 时序扩展 + 11 截面均值 + Barra 风格因子），LightGBM 回归拟合「日截面 `rank(pct)-0.5`」标签。诚实 OOS（train ≤ 截止日，无早停泄露）：

| 窗口 | 训练截止 | 评估区间 | RankIC | IR |
|------|----------|----------|--------|-----|
| 主窗 | 2021-12-31 | 2022 全年 | 0.0508 | 1.734 |
| 副窗 | 2021-06-30 | 2021 下半年 | 0.0441 | 2.144 |

```sh
python submission/train.py --data-dir <数据根目录> --out submission/model
```

推理（`submission/model/lgb_reg.txt`）：

```python
from submission.inference import InferenceModel

model = InferenceModel("submission/model")
scores = model.predict(as_of_date="2022-01-04", data=data)
```

## 冠军 FNR 方案复刻

`champion/` 为 `Star6azeR/CSI300-Challenge-2026` 冠军方案的完整复刻（仅移除冻结权重，代码保留）。核心架构：

- **347 维 mixed61 特征**：Barra 风格因子 + 量价因子 + 日频时序/窗口统计特征。
- **冻结底座**：LightGBM + TabM 集成，逐日截面 rank 打分。
- **截面自注意力教师**（347→32，4 头）：在同交易日股票间建模，用 rank MSE 训练。
- **kNN 邻域头**：固定图、无学习注意力的成对排序（按收益差加权封顶）。
- **尾部重排输出层**：保留底座前 60 名及其顺序，其余按邻域头增强分数重排。

### 复刻层改动（相对原仓库）

为在 Windows + 当前依赖版本下可复现，做了以下工程改动，未触及算法语义：

- `fnr/training/peer.py`：`resource` 模块改为可选导入（Windows 无此模块）。
- `fnr/training/base_training.py`：支持截断的「时点」数据树（缺失的后续月份跳过）。
- `fnr/inference/*`、`fnr/training/artifacts.py`、`export.py`、`kernels.py`：路径统一 `as_posix()`，校验和（sha256）对文本文件做 CRLF 归一化，保证跨平台一致。
- `fnr/training/ple/deep_learner.py`：新增 `FNR_TORCH_THREADS` 环境变量控制 CPU 线程数（默认 2），并放宽 `n_jobs==2` 硬编码，便于 CPU 加速复现。

训练依赖（PyTorch CPU、TabM、rtdl-num-embeddings）见 `champion/requirements-training.txt`。

```sh
python -m pip install -r champion/requirements-training.txt
python -m champion.train --data-dir <数据根目录> --model-dir <输出目录>
```

数据布局为 `表名/年/月/YYYYMM.parquet`（`MarketData`、`Factor33_winsor`、`Barra_factor`）。

## 双窗口 OOS 验证与实验记录

`scripts/dual_window_validate.py` 固化「双窗口门槛」：主窗 2022 全年、副窗 2021H2；改动须满足主窗 RankIC 增益 ≥0.003 且 IR 不降、副窗不恶化，否则否决。

在此门槛下，逐条隔离验证了 5 条冠军技巧，**全部否决**：

| # | 技巧 | 结果 |
|---|------|------|
| 1 | 尾部重排输出层 | 无增强分数时为恒等，无增益 |
| 2 | lambdarank 全局成对 | 主窗 0.0437、副窗 IR 崩至 1.226 |
| 3 | naive 邻域成对排序 | 增强分数仅 0.023，弱于基线 |
| 4 | 截面自注意力教师（71→32→1） | RankIC 0.0375，未学到 |
| 5 | rank(pct) 特征预处理 | 主窗 0.0466、副窗 IR 崩至 0.457 |

失败根因：冠军的成对排序是**邻域内局部**、注意力是给 **347 维 + TabM 底座**使用的；孤立拆到 71 维单 LightGBM 上全部劣化。因此复刻方向为完整 347 维 + TabM 底座，而非孤立拆解。相关实验脚本见 `scripts/exp2_*`~`exp5_*`、`scripts/fnr_*`。

## 安装

基线（推理）依赖见 `submission/requirements.txt`；冠军复刻训练依赖见 `champion/requirements-training.txt`。

```sh
python -m pytest tests/
```

## 归属

`champion/` 代码源自 [Star6azeR/CSI300-Challenge-2026](https://github.com/Star6azeR/CSI300-Challenge-2026)，其原始说明保留在 [champion/ATTRIBUTION.md](champion/ATTRIBUTION.md)。本项目仅作学习复刻与研究，不构成任何投资建议。

## License

[Apache-2.0](LICENSE)
