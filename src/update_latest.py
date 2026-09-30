"""Fetch latest TXO data and publish VIX-style plus TX/MTX IV charts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from market_activity import generate_front_vix_taiex_chart
from option_volume import (
    calculate_expiration_volume,
    update_volume_history,
)
from txo_iv import update_iv_outputs
from txo_vix import (
    find_latest_available_term_structure,
    get_taifex_options_data,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

HISTORY_PATH = (
    DATA_DIR / "vix_term_structure_history.csv"
)

OPTION_VOLUME_HISTORY_PATH = (
    DATA_DIR / "option_volume_history.csv"
)

PNG_PATH = (
    OUTPUT_DIR / "latest_term_structure.png"
)

SVG_PATH = (
    OUTPUT_DIR / "latest_term_structure.svg"
)

JSON_PATH = (
    OUTPUT_DIR / "latest.json"
)


# ============================================================
# Candidate trading dates
# ============================================================

def candidate_dates(
    lookback_days: int = 10,
) -> list[pd.Timestamp]:

    today_taipei = (
        pd.Timestamp.now(
            tz="Asia/Taipei"
        )
        .normalize()
        .tz_localize(None)
    )

    return [
        today_taipei
        - pd.Timedelta(
            offset,
            unit="D",
        )
        for offset in range(
            lookback_days
        )
    ]


# ============================================================
# VIX history
# ============================================================

def update_history(
    latest: pd.DataFrame,
) -> pd.DataFrame:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if HISTORY_PATH.exists():

        history = pd.read_csv(
            HISTORY_PATH
        )

    else:

        history = pd.DataFrame(
            columns=latest.columns
        )

    combined = pd.concat(
        [
            history,
            latest,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "Date",
                "Expiration_Month",
            ],
            keep="last",
        )
        .sort_values(
            [
                "Date",
                "Days_to_Exp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    combined.to_csv(
        HISTORY_PATH,
        index=False,
    )

    return combined


# ============================================================
# Static VIX term-structure chart
# ============================================================

def draw_chart(
    latest: pd.DataFrame,
    as_of_date: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6.4)
    )

    axis.plot(
        latest["Days_to_Exp"],
        latest["VIX"],
        marker="o",
        linewidth=2.2,
    )

    for row in latest.itertuples(
        index=False
    ):

        axis.annotate(
            str(
                row.Expiration_Month
            ),
            (
                row.Days_to_Exp,
                row.VIX,
            ),
            xytext=(
                5,
                7,
            ),
            textcoords=
                "offset points",
            fontsize=8,
        )

    axis.set_title(
        "TXO VIX-style "
        "Implied-Volatility "
        f"Term Structure — {as_of_date}"
    )

    axis.set_xlabel(
        "Days to expiration"
    )

    axis.set_ylabel(
        "Annualized implied "
        "volatility (%)"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.margins(
        x=0.04,
        y=0.12,
    )

    note = (
        "Research estimate from "
        "end-of-day TXO quotes; "
        "not the official TAIWAN VIX. "
        "Risk-free rate and sampling "
        "rules are documented in "
        "the repository."
    )

    figure.text(
        0.5,
        0.01,
        note,
        ha="center",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(
            0,
            0.04,
            1,
            1,
        )
    )

    figure.savefig(
        PNG_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    figure.savefig(
        SVG_PATH,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# Metadata
# ============================================================

def write_metadata(
    latest: pd.DataFrame,
    as_of_date: str,
) -> None:

    metadata = {

        "as_of_date":
            as_of_date,

        "generated_at_taipei":
            pd.Timestamp.now(
                tz="Asia/Taipei"
            ).isoformat(
                timespec="seconds"
            ),

        "timezone":
            "Asia/Taipei",

        "number_of_expirations":
            int(
                len(latest)
            ),

        "minimum_dte":
            float(
                latest[
                    "Days_to_Exp"
                ].min()
            ),

        "maximum_dte":
            float(
                latest[
                    "Days_to_Exp"
                ].max()
            ),

        "risk_free_rate":
            float(
                os.getenv(
                    "RISK_FREE_RATE",
                    "0.01",
                )
            ),

        "official_index":
            False,
    }

    JSON_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# TXO option volume by expiration
# ============================================================

def update_option_volume(
    as_of: pd.Timestamp,
) -> pd.DataFrame:

    raw_options = (
        get_taifex_options_data(
            pd.Timestamp(
                as_of
            )
        )
    )

    latest_volume = (
        calculate_expiration_volume(
            raw_options,
            as_of,
        )
    )

    update_volume_history(
        latest_volume,
        OPTION_VOLUME_HISTORY_PATH,
    )

    return latest_volume


# ============================================================
# Main
# ============================================================

def main() -> None:

    risk_free_rate = float(
        os.getenv(
            "RISK_FREE_RATE",
            "0.01",
        )
    )

    # --------------------------------------------------------
    # Find latest available VIX term structure
    # --------------------------------------------------------

    as_of, latest = (
        find_latest_available_term_structure(
            candidate_dates(),
            risk_free_rate=
                risk_free_rate,
        )
    )

    as_of_text = (
        as_of.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # Update VIX history
    # --------------------------------------------------------

    update_history(
        latest
    )

    # --------------------------------------------------------
    # Static VIX term-structure outputs
    # --------------------------------------------------------

    draw_chart(
        latest,
        as_of_text,
    )

    write_metadata(
        latest,
        as_of_text,
    )

    # --------------------------------------------------------
    # TXO option volume by expiration
    # --------------------------------------------------------

    latest_volume = None

    try:

        latest_volume = (
            update_option_volume(
                as_of
            )
        )

        print(
            "\n"
            "TXO option volume "
            "by expiration:"
        )

        print(
            latest_volume.to_string(
                index=False
            )
        )

    except Exception as exc:

        print(
            "\nWARNING: could not "
            "update TXO option volume: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # TAIEX vs front-expiry VIX
    # --------------------------------------------------------

    try:

        generate_front_vix_taiex_chart(
            history_path=
                HISTORY_PATH,

            output_path=
                OUTPUT_DIR
                / "latest_front_vix_vs_taiex.png",

            window_days=30,
        )

    except Exception as exc:

        print(
            "\nWARNING: could not "
            "generate TAIEX/"
            "front-expiry VIX chart: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # TX / MTX IV curves
    # --------------------------------------------------------

    iv_summary = None

    try:

        iv_summary = (
            update_iv_outputs(
                trade_date=
                    as_of,

                output_dir=
                    OUTPUT_DIR,

                data_dir=
                    DATA_DIR,

                risk_free_rate=
                    risk_free_rate,
            )
        )

    except Exception as exc:

        print(
            "\nWARNING: could not "
            "update TX / MTX IV "
            f"outputs for {as_of_text}: "
            f"{exc}"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        "\nVIX term structure:"
    )

    print(
        latest.to_string(
            index=False
        )
    )

    print(
        "\nUpdated VIX-style "
        f"chart for {as_of_text}: "
        f"{PNG_PATH}"
    )

    print(
        "Updated VIX history: "
        f"{HISTORY_PATH}"
    )

    if latest_volume is not None:

        print(
            "Updated option-volume "
            "history: "
            f"{OPTION_VOLUME_HISTORY_PATH}"
        )

    else:

        print(
            "Option-volume history "
            "was not updated "
            "on this run."
        )

    if iv_summary is not None:

        print(
            "Updated IV calibration "
            f"outputs: {iv_summary}"
        )

    else:

        print(
            "TX / MTX IV outputs "
            "were not updated "
            "on this run."
        )


if __name__ == "__main__":
    main()