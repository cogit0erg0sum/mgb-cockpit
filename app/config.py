# app/config.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
ENGINE_DIR = ROOT / "engine"
DATA = APP_DIR / "data"
DASH = DATA / "dashboard.csv"
WL   = DATA / "watchlist.csv"
FUND = DATA / "fundamentals.csv"
META = DATA / "metadata.json"

for p in (APP_DIR, ENGINE_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)