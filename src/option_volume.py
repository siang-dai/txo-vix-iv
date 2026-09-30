from __future__ import annotations

from pathlib import Path

import pandas as pd


def calculate_expiration_volume(
    raw_options: pd.DataFrame,
    trade_date,
) -> pd.DataFrame:
    """
    Aggregate TXO option volume by expiration.

    Volume =
        sum of all strikes
        + calls
        + puts
        for the same expiration.
    """

    df = raw_options.copy()

    required = {
        "契約",
        "到期月份(週別)",
        "成交量",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required option-volume columns: "
            + ", ".join(sorted(missing))
        )

    # TXO only
    df["契約"] = (
        df["契約"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["契約"] == "TXO"
    ].copy()

    # Clean expiration label
    df["到期月份(週別)"] = (
        df["到期月份(週別)"]
        .astype(str)
        .str.strip()
    )

    # Volume may contain commas / non-numeric values
    df["成交量"] = pd.to_numeric(
        df["成交量"]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)

    result = (
        df.groupby(
            "到期月份(週別)",
            as_index=False,
        )["成交量"]
        .sum()
        .rename(
            columns={
                "到期月份(週別)": "Expiration_Month",
                "成交量": "Volume",
            }
        )
    )

    result.insert(
        0,
        "Date",
        pd.Timestamp(trade_date).strftime(
            "%Y-%m-%d"
        ),
    )

    result["Volume"] = (
        result["Volume"]
        .round()
        .astype(int)
    )

    return result.sort_values(
        "Expiration_Month"
    ).reset_index(drop=True)


def update_volume_history(
    latest: pd.DataFrame,
    history_path: str | Path,
) -> pd.DataFrame:
    """
    Append today's expiration-volume observations
    while preserving older dates.
    """

    history_path = Path(history_path)

    if history_path.exists():
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame(
            columns=[
                "Date",
                "Expiration_Month",
                "Volume",
            ]
        )

    combined = pd.concat(
        [history, latest],
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
                "Expiration_Month",
            ]
        )
        .reset_index(drop=True)
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        history_path,
        index=False,
    )

    return combined