# engine/auto_refresh.py
import pathlib
from datetime import datetime
import pandas as pd

from screener_core import compute_dashboard
from fundamentals_core import compute_fundamentals

ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root
APP = ROOT / "app"
DATA = APP / "data"
WL   = DATA / "watchlist.csv"
DASH = DATA / "dashboard.csv"
FUND = DATA / "fundamentals.csv"

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if not WL.exists():
        pd.DataFrame(columns=["symbol","name","segment","buy_low","buy_high","stop","notes"]).to_csv(WL, index=False)

    # 1) Screener (technical signals)
    n1 = compute_dashboard(str(WL), str(DASH), days=420)

    # 2) Fundamentals (Yahoo free, delayed)
    n2 = compute_fundamentals(str(WL), str(FUND), throttle_sec=0.4)

    print(f"[{datetime.now().isoformat()}] Refreshed: signals={n1}, fundamentals={n2}")
    print(f"Updated: {DASH} and {FUND}")

if __name__ == "__main__":
    main()