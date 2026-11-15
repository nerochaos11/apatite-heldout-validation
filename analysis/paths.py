from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "02_原始数据与来源"
OUT = Path(os.environ.get("APATITE_RESULTS_DIR", ROOT / "01_数据与结果"))
MODEL_DIR = Path(os.environ.get("APATITE_APPLICATION_DIR", ROOT / "06_模型与地图"))
MAP_DIR = MODEL_DIR / "交互地图"
FIGURE_DIR = Path(os.environ.get("APATITE_FIGURE_DIR", ROOT / "再计算/图件"))
