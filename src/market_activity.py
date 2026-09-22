from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json


TWSE_TAIEX_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"


def _roc_date_to_timestamp(value: object) -> pd.Timestamp:
    """Convert a TWSE ROC-calendar date such as 115/09/18 to Timestamp."""
    text = str(value).strip()
    parts = text.split("/")

    if len(parts) != 3:
        return pd.NaT

    year = int(parts[0])
    if year < 1911:
        year += 1911

    return pd.Timestamp(
        year=year,
        month=int(parts[1]),
        day=int(parts[2]),
    )


def _get_taiex_month(
    year: int,
    month: int,
    session: requests.Session,
) -> pd.DataFrame:
    request_date = f"{year:04d}{month:02d}01"

    response = session.get(
        TWSE_TAIEX_URL,
        params={
            "response": "json",
            "date": request_date,
        },
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    rows = payload.get("data", [])

    output: list[dict[str, object]] = []

    for row in rows:
        if len(row) < 5:
            continue

        date = _roc_date_to_timestamp(row[0])
        close = pd.to_numeric(
            str(row[4]).replace(",", ""),
            errors="coerce",
        )

        if pd.isna(date) or pd.isna(close):
            continue

        output.append(
            {
                "Date": date,
                "TAIEX_Close": float(close),
            }
        )

    return pd.DataFrame(
        output,
        columns=["Date", "TAIEX_Close"],
    )


def _get_taiex_history(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    session: requests.Session,
) -> pd.DataFrame:
    months = pd.period_range(
        start=start_date,
        end=end_date,
        freq="M",
    )

    frames = [
        _get_taiex_month(period.year, period.month, session)
        for period in months
    ]

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(columns=["Date", "TAIEX_Close"])

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _get_front_vix(
    history: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    required = {
        "Date",
        "Expiration_Month",
        "Expiration_Date",
        "Days_to_Exp",
        "VIX",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "VIX history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = history.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Days_to_Exp"] = pd.to_numeric(
        df["Days_to_Exp"],
        errors="coerce",
    )
    df["VIX"] = pd.to_numeric(
        df["VIX"],
        errors="coerce",
    )

    df = df[
        df["Date"].notna()
        & df["Days_to_Exp"].notna()
        & df["VIX"].notna()
        & (df["Days_to_Exp"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid VIX history rows are available.")

    # Nearest available expiry for each trading date.
    front = (
        df.sort_values(["Date", "Days_to_Exp", "Expiration_Date"])
        .groupby("Date", as_index=False)
        .first()
    )

    front = front[
        [
            "Date",
            "Expiration_Month",
            "Expiration_Date",
            "Days_to_Exp",
            "VIX",
        ]
    ].rename(
        columns={
            "Expiration_Month": "Contract",
            "VIX": "Front_VIX",
        }
    )

    latest = front["Date"].max()
    cutoff = latest - pd.Timedelta(days=window_days - 1)

    return (
        front[front["Date"] >= cutoff]
        .sort_values("Date")
        .reset_index(drop=True)
    )


def generate_front_vix_taiex_chart(
    history_path: str | Path,
    output_path: str | Path,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Plot TAIEX close against the nearest-expiry VIX-style IV.

    The function reads the existing VIX term-structure history, downloads
    only the TWSE monthly index-history JSON needed for the latest window,
    and writes one PNG chart for the website.

    Returns the merged plotting DataFrame for diagnostics/tests.
    """
    history_path = Path(history_path)
    output_path = Path(output_path)

    history = pd.read_csv(history_path)
    front = _get_front_vix(history, window_days=window_days)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "Chrome/140 Safari/537.36"
            ),
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }
    )

    taiex = _get_taiex_history(
        start_date=front["Date"].min(),
        end_date=front["Date"].max(),
        session=session,
    )

    data = (
        front.merge(taiex, on="Date", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    plot_data = data.dropna(
        subset=["TAIEX_Close", "Front_VIX"]
    ).copy()

    if len(plot_data) < 2:
        raise RuntimeError(
            "Not enough overlapping TAIEX/VIX observations to draw chart."
        )

    fig, (ax_index, ax_vix) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.6),
        sharex=True,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 上圖：TAIEX
    ax_index.plot(
        plot_data["Date"],
        plot_data["TAIEX_Close"],
        marker="o",
        linewidth=2,
        markersize=4,
        color=colors[0],
        label="TAIEX Close",
    )

    ax_index.set_ylabel("TAIEX Close")
    ax_index.set_title("TAIEX vs Front-expiry VIX-style IV\nLast 30 Calendar Days")
    ax_index.grid(axis="y", alpha=0.2)
    ax_index.legend(loc="upper left", frameon=False)

    # 下圖：Front-expiry VIX
    ax_vix.plot(
        plot_data["Date"],
        plot_data["Front_VIX"],
        marker="s",
        linewidth=2,
        markersize=4,
        color=colors[1],
        label="Front-expiry VIX-style IV",
    )

    ax_vix.set_ylabel("VIX-style IV (%)")
    ax_vix.set_xlabel("Trading Date")
    ax_vix.grid(axis="y", alpha=0.2)
    ax_vix.legend(loc="upper left", frameon=False)

    # 共用 x 軸格式
    ax_vix.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_vix.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_vix.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Small JSON payload for the interactive website chart
    # ------------------------------------------------------------

    json_path = output_path.with_suffix(".json")

    interactive_data = plot_data[
        [
            "Date",
            "Contract",
            "TAIEX_Close",
            "Front_VIX",
        ]
    ].copy()

    interactive_data["Date"] = (
        interactive_data["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    payload = {
        "as_of_date": interactive_data["Date"].iloc[-1],
        "window_days": window_days,
        "data": interactive_data.to_dict(orient="records"),
    }

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return data
