<p align="center"><strong>⚠️ 本仓库内容仅供学习研究用 ⚠️</strong></p>
<p align="center"><strong>🚫 禁止以任何方式应用于实盘 🚫</strong></p>

# 沪深300指数增强与截面选股方案

基于 LightGBM 的沪深300截面排序模型，针对「截面选股」评测口径（次日 open-to-open 收益的截面 RankIC）训练。模型以公开 2016–2022 年数据训练，推理时对每个信号日输出成分股分数 `score`。

本仓库为参赛方案源码整理版，供学习与研究参考，不构成任何投资建议。

## 安装

依赖已锁定版本，见 `submission/requirements.txt`。建议 Python 3.9+（开发环境 3.13）：

```sh
python -m pip install -r submission/requirements.txt
```

## 运行

### 训练（可选，从公开数据复现模型）

准备 2016–2022 年的 `MarketData`、`Factor33_winsor`、`Barra_factor` 原始数据，指定数据根目录与模型输出目录：

```sh
python submission/train.py --data-dir <数据根目录> --out submission/model
```

### 推理

加载随包权重 `submission/model/lgb_reg.txt`，对信号日输出 `date/code/score`：

```python
from submission.inference import InferenceModel

model = InferenceModel("submission/model")
scores = model.predict(as_of_date="2022-01-04", data=data)
```

`as_of_date` 为信号日；`data` 为截至该日 120 个交易日的上述三张表字典。

### 测试

```sh
python -m pytest tests/
```

## 目录结构

```
├── submission/            # 提交包：推理入口、训练代码、依赖与冻结权重
│   ├── inference.py       # 推理入口（InferenceModel，自包含特征构建）
│   ├── train.py           # 训练脚本（可复现 model/）
│   ├── requirements.txt   # 锁版依赖
│   └── model/             # 冻结权重 + 特征元数据
├── src/                   # 开发期代码（清洗/特征/数据加载/验证/训练）
├── scripts/               # 实验与验证脚本
├── notebooks/             # 探索性分析
└── tests/                 # 最小 smoke test
```

> 比赛原始数据（`data.zip`、`data_temp/`）、回测结果（`result/`）与中间产物不入库，请按赛事授权自行获取数据。

## License

[Apache-2.0](LICENSE)
