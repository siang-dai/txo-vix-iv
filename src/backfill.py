"""Backfill historical TXO VIX-style term structures from TAIFEX.

Run this locally rather than as a daily GitHub Actions job. The script resumes
from the existing CSV and sleeps between requests to reduce load on TAIFEX.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from txo_vix import (
    TaifexDownloadError,
    calculate_term_structure,
    get_taifex_options_data,
    preprocess_txo_data,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "data" / "vix_term_structure_history.csv"


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


def save_history(history: pd.DataFrame) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = history.drop_duplicates(
        subset=["Date", "Expiration_Month"], keep="last"
    ).sort_values(["Date", "Days_to_Exp"])
    history.to_csv(HISTORY_PATH, index=False)


def main() -> None:
    args = parse_args()
    dates = pd.date_range(args.start, args.end, freq="B")
    history = load_history()
    completed_dates = set() if args.overwrite or history.empty else set(history["Date"].astype(str))

    for trading_date in dates:
        date_text = trading_date.strftime("%Y-%m-%d")
        if date_text in completed_dates:
            print(f"SKIP {date_text}: already in history")
            continue

        try:
            raw = get_taifex_options_data(trading_date)
            if raw is None or raw.empty:
                print(f"EMPTY {date_text}")
            else:
                cleaned = preprocess_txo_data(raw)
                result = calculate_term_structure(
                    cleaned,
                    as_of=trading_date,
                    risk_free_rate=args.risk_free_rate,
                )
                if result.empty:
                    print(f"NO RESULT {date_text}")
                else:
                    history = pd.concat([history, result], ignore_index=True)
                    save_history(history)
                    print(f"OK {date_text}: {len(result)} expirations")
        except (TaifexDownloadError, ValueError, KeyError) as exc:
            print(f"ERROR {date_text}: {exc}")

        time.sleep(max(args.sleep, 0.0))


if __name__ == "__main__":
    main()
