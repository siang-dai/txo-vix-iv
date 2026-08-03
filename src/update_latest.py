"""Fetch the latest available TXO data, update history, and draw a static chart."""

from __future__ import annotations

import json
import os
from datetime import timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from txo_vix import find_latest_available_term_structure

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
HISTORY_PATH = DATA_DIR / "vix_term_structure_history.csv"
PNG_PATH = OUTPUT_DIR / "latest_term_structure.png"
SVG_PATH = OUTPUT_DIR / "latest_term_structure.svg"
JSON_PATH = OUTPUT_DIR / "latest.json"


def candidate_dates(lookback_days: int = 10) -> list[pd.Timestamp]:
    today_taipei = pd.Timestamp.now(tz="Asia/Taipei").normalize().tz_localize(None)
    return [today_taipei - pd.Timedelta(days=offset) for offset in range(lookback_days)]


def update_history(latest: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
    else:
        history = pd.DataFrame(columns=latest.columns)

    combined = pd.concat([history, latest], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["Date", "Expiration_Month"], keep="last"
    ).sort_values(["Date", "Days_to_Exp"])
    combined.to_csv(HISTORY_PATH, index=False)
    return combined


def draw_chart(latest: pd.DataFrame, as_of_date: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6.4))
    axis.plot(latest["Days_to_Exp"], latest["VIX"], marker="o", linewidth=2.2)

    for row in latest.itertuples(index=False):
        axis.annotate(
            str(row.Expiration_Month),
            (row.Days_to_Exp, row.VIX),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_title(f"TXO VIX-style Implied-Volatility Term Structure — {as_of_date}")
    axis.set_xlabel("Days to expiration")
    axis.set_ylabel("Annualized implied volatility (%)")
    axis.grid(True, alpha=0.25)
    axis.margins(x=0.04, y=0.12)

    note = (
        "Research estimate from end-of-day TXO quotes; not the official TAIWAN VIX. "
        "Risk-free rate and sampling rules are documented in the repository."
    )
    figure.text(0.5, 0.01, note, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    figure.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(figure)


def write_metadata(latest: pd.DataFrame, as_of_date: str) -> None:
    metadata = {
        "as_of_date": as_of_date,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "number_of_expirations": int(len(latest)),
        "minimum_dte": float(latest["Days_to_Exp"].min()),
        "maximum_dte": float(latest["Days_to_Exp"].max()),
        "risk_free_rate": float(os.getenv("RISK_FREE_RATE", "0.01")),
        "official_index": False,
    }
    JSON_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.01"))
    as_of, latest = find_latest_available_term_structure(
        candidate_dates(),
        risk_free_rate=risk_free_rate,
    )
    as_of_text = as_of.strftime("%Y-%m-%d")
    update_history(latest)
    draw_chart(latest, as_of_text)
    write_metadata(latest, as_of_text)
    print(latest.to_string(index=False))
    print(f"Updated chart for {as_of_text}: {PNG_PATH}")


if __name__ == "__main__":
    main()
