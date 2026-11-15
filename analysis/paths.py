from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "02_source_data"
OUT = Path(os.environ.get("APATITE_RESULTS_DIR", ROOT / "01_data_and_results"))
MODEL_DIR = Path(os.environ.get("APATITE_APPLICATION_DIR", ROOT / "06_model_and_map"))
MAP_DIR = MODEL_DIR / "interactive_map"
FIGURE_DIR = Path(os.environ.get("APATITE_FIGURE_DIR", ROOT / "recomputed/figures"))
