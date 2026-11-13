# 提交包说明

本目录是竞赛提交的完整代码与冻结权重，训练与推理代码均自包含（不依赖 `src/` 或外部模块），可直接用于复现与评测。

## 目录结构

```
submission/
├── inference.py       # 推理入口：InferenceModel 类，内联特征构建
├── train.py           # 训练脚本：从公开数据复现 model/
├── requirements.txt   # 锁版依赖（pandas/numpy/lightgbm/pyarrow）
└── model/
    ├── lgb_reg.txt    # LightGBM 冻结权重
    └── metadata.json  # 特征列清单、标签列、seed、best_iter、训练区间
```

## 训练

```sh
python -m pip install -r requirements.txt
python train.py --data-dir <数据根目录> --out model [--seed 42]
```

`--data-dir` 下需包含三个子目录，各含按年月切分的 `.parquet`：

| 表 | 关键列 |
| --- | --- |
| `MarketData` | 日期、代码、open/close/high/low/amount |
| `Factor33_winsor` | 日期、代码、33 个缩尾因子 |
| `Barra_factor` | 日期、代码、10 个 Barra 风格因子（中文列名） |

标签采用「次日 O2O 收益的截面百分位 − 0.5」（`target_rank`），与评测 RankIC 口径对齐。

## 推理

```python
from inference import InferenceModel

model = InferenceModel("model")
df = model.predict(as_of_date="2022-01-04", data=data)
# df 列：date / code / score
```

- `as_of_date`：信号日，`"YYYY-MM-DD"` 字符串。
- `data`：截至该日 120 个交易日的 `{"MarketData": ..., "Factor33_winsor": ..., "Barra_factor": ...}` 字典。
- 输出覆盖当日 Barra 成分股全集，`score` 为截面排序分数。
- 推理过程不重新拟合、不读包外数据、不联网。
