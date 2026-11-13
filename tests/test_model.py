"""提交包最小 smoke test：验证模型权重可加载、元数据自洽、依赖已锁版。

运行：python -m pytest tests/
"""
import json
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "submission" / "model"
META_PATH = MODEL_DIR / "metadata.json"
REQ_PATH = ROOT / "submission" / "requirements.txt"


def _load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def test_metadata_is_valid():
    meta = _load_meta()
    assert isinstance(meta["feature_columns"], list)
    assert len(meta["feature_columns"]) > 0
    assert meta["label_col"] == "target_rank"
    assert meta["best_iter"] > 0


def test_model_loads_and_matches_metadata():
    meta = _load_meta()
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb_reg.txt"))
    assert booster.num_feature() == len(meta["feature_columns"])


def test_requirements_are_pinned():
    text = REQ_PATH.read_text(encoding="utf-8")
    pins = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert pins, "requirements.txt 为空"
    for pin in pins:
        assert "==" in pin, f"依赖未锁定版本: {pin}"
