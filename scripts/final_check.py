"""最终自检：模拟平台加载 submission 包，验证接口与输出格式"""
import sys, importlib.util
from pathlib import Path
import pandas as pd

sub_dir = Path("submission").resolve()
sys.path.insert(0, str(sub_dir))
# 清除可能的模块缓存
for m in list(sys.modules.keys()):
    if m in ("features", "inference", "sub_inference"):
        del sys.modules[m]
spec = importlib.util.spec_from_file_location("sub_inference", sub_dir / "inference.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["sub_inference"] = mod
spec.loader.exec_module(mod)

# 模拟平台调用
model = mod.InferenceModel(str(sub_dir / "model"))

sys.path.insert(0, "src")
from data_loader import DataLoader
dl = DataLoader("data_temp/Data")
md = dl.load_market_data()
f33 = dl.load_factor33()
bar = dl.load_barra()

d = 20220601
cal = sorted(md["日期"].unique())
idx = cal.index(d)
window = cal[max(0, idx-119): idx+1]
data = {
    "MarketData": md.loc[md["日期"].isin(window)],
    "Factor33_winsor": f33.loc[f33["日期"].isin(window)],
    "Barra_factor": bar.loc[bar["日期"].isin(window)],
}
pred = model.predict("2022-06-01", data)

# 自检清单
checks = []
checks.append(("输出三列 date/code/score", list(pred.columns) == ["date", "code", "score"]))
checks.append(("date 格式 YYYY-MM-DD", pred["date"].iloc[0] == "2022-06-01"))
checks.append(("score 为 float 且有限", pd.api.types.is_float_dtype(pred["score"]) and pred["score"].notna().all()))
checks.append(("覆盖当日 Barra 全池300只", len(pred) == 300))
bar_codes = set(bar.loc[bar["日期"].eq(d), "代码"])
checks.append(("code 与 Barra 当日完全一致", set(pred["code"]) == bar_codes))
checks.append(("无重复 code", pred["code"].is_unique))

print("=" * 50)
for name, ok in checks:
    print(f"  {'✅' if ok else '❌'} {name}")
print("=" * 50)
assert all(ok for _, ok in checks), "自检未通过"
print("✅ 全部自检通过，submission/ 可提交")
