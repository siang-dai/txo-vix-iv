"""Backfill historical TXO VIX-style term structures from TAIFEX.

Run this locally rather than as a daily GitHub Actions job. The script resumes
from the existing CSV and sleeps between requests to reduce load on TAIFEX.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from option_volume import calculate_expiration_volume, update_volume_history
from txo_vix import (
    TaifexDownloadError,
    calculate_term_structure,
    get_taifex_options_data,
    preprocess_txo_data,
    replace_trading_day,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "data" / "vix_term_structure_history.csv"
VOLUME_PATH = ROOT / "data" / "option_volume_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date, for example 2020-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between requests")
    parser.add_argument("--risk-free-rate", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    dates = pd.date_range(args.start, args.end, freq="B")
    history = load_history()
    vix_dates = set(history["Date"]) if not history.empty else set()
    volume = pd.read_csv(VOLUME_PATH) if VOLUME_PATH.exists() else pd.DataFrame()
    fields = ["Call_Volume", "Put_Volume", "Call_OI", "Put_OI", "Expiration_Date",
              "First_Strike_Bin", "Call_Bins_500", "Put_Bins_500"]
    volume_dates = set()
    if set(fields).issubset(volume):
        complete = volume[fields].notna().all(axis=1).groupby(volume["Date"]).all()
        volume_dates = set(complete.index[complete])

    for trading_date in dates:
        date_text = trading_date.strftime("%Y-%m-%d")
        if not args.overwrite and date_text in vix_dates and date_text in volume_dates:
            print(f"SKIP {date_text}: already in history")
            continue

        try:
            raw = get_taifex_options_data(trading_date)
            if raw is None or raw.empty:
                print(f"EMPTY {date_text}")
            else:
                if args.overwrite or date_text not in vix_dates:
                    result = calculate_term_structure(
                        preprocess_txo_data(raw), as_of=trading_date, risk_free_rate=args.risk_free_rate
                    )
                    history = replace_trading_day(history, date_text, result)
                    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
                    history.to_csv(HISTORY_PATH, index=False)
                    print(f"OK {date_text}: {len(result)} VIX expirations")
                if args.overwrite or date_text not in volume_dates:
                    activity = calculate_expiration_volume(raw, trading_date)
                    update_volume_history(activity, VOLUME_PATH)
                    print(f"OK {date_text}: {len(activity)} volume expirations")
        except (TaifexDownloadError, ValueError, KeyError) as exc:
            print(f"ERROR {date_text}: {exc}")

        time.sleep(max(args.sleep, 0.0))


if __name__ == "__main__":
    main()
