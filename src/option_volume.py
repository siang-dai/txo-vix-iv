from __future__ import annotations

from pathlib import Path

import pandas as pd


def calculate_expiration_volume(raw_options: pd.DataFrame, trade_date) -> pd.DataFrame:
    """Aggregate all-session volume and post-settlement open interest by expiry and side."""
    day = pd.Timestamp(trade_date).normalize()
    rows = raw_options.loc[raw_options["契約"].str.strip().eq("TXO")].copy()
    if rows.empty or not pd.to_datetime(rows["交易日期"]).eq(day).all():
        raise ValueError(f"Expected TXO data for {day.date()}")

    rows["Expiration_Month"] = rows["到期月份(週別)"].str.strip()
    rows["Expiration_Date"] = pd.to_datetime(rows["契約到期日"].astype(str), format="%Y%m%d")
    rows["Side"] = rows["買賣權"].str.strip().map({"買權": "Call", "賣權": "Put"})
    if rows[["Side", "Expiration_Date", "Expiration_Month"]].isna().any().any():
        raise ValueError("Incomplete option identifiers")
    rows["Volume"] = pd.to_numeric(rows["成交量"].astype(str).str.replace(",", "", regex=False))
    regular = rows["交易時段"].str.strip().eq("一般")
    if not regular.any():
        raise ValueError("Regular-session open interest is unavailable")
    # TAIFEX includes expiry-day trades, but settled contracts have no closing OI.
    open_positions = regular & rows["Expiration_Date"].gt(day)
    rows["OI"] = 0
    rows.loc[open_positions, "OI"] = pd.to_numeric(
        rows.loc[open_positions, "未沖銷契約數"].astype(str).str.replace(",", "", regex=False)
    )
    if rows[["Volume", "OI"]].isna().any().any() or (rows[["Volume", "OI"]] < 0).any().any():
        raise ValueError("Missing or negative contract counts")

    totals = rows.groupby(["Expiration_Month", "Expiration_Date", "Side"])[["Volume", "OI"]].sum()
    by_side = totals.unstack("Side", fill_value=0).reindex(
        columns=pd.MultiIndex.from_product([["Volume", "OI"], ["Call", "Put"]]), fill_value=0
    )
    by_side.columns = [f"{side}_{measure}" for measure, side in by_side.columns]
    result = by_side.reset_index().sort_values(["Expiration_Date", "Expiration_Month"])
    result["Expiration_Date"] = result["Expiration_Date"].dt.strftime("%Y-%m-%d")
    result.insert(0, "Date", day.strftime("%Y-%m-%d"))
    result.insert(2, "Volume", result["Call_Volume"] + result["Put_Volume"])
    return result.reset_index(drop=True)


def update_volume_history(latest: pd.DataFrame, history_path: str | Path) -> pd.DataFrame:
    """Replace complete sessions, keeping all other trading dates."""
    path = Path(history_path)
    combined = latest
    if path.exists():
        history = pd.read_csv(path)
        remaining = history.loc[~history.Date.isin(latest.Date)]
        if not remaining.empty:
            combined = pd.concat([remaining, latest], ignore_index=True)
    combined = combined.sort_values(["Date", "Expiration_Date", "Expiration_Month"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
