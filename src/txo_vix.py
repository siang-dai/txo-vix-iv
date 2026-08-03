"""TXO VIX-style term-structure utilities.

This module intentionally labels the result "VIX-style" rather than the official
TAIWAN VIX.  It estimates a variance for each listed expiration using end-of-day
TXO quotes.  The official exchange index includes additional sampling filters,
time-to-expiration conventions, and a 30-day interpolation procedure.
"""

from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd
import requests

TAIFEX_URL = "https://www.taifex.com.tw/cht/3/optDataDown"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)


class TaifexDownloadError(RuntimeError):
    """Raised when TAIFEX data cannot be downloaded or parsed."""


@dataclass(frozen=True)
class DownloadConfig:
    timeout_seconds: int = 30
    max_attempts: int = 3
    retry_wait_seconds: float = 3.0


def _format_query_date(value: date | datetime | pd.Timestamp | str) -> str:
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="raise")
    else:
        parsed = pd.Timestamp(value)
    return parsed.strftime("%Y/%m/%d")


def get_taifex_options_data(
    trading_date: date | datetime | pd.Timestamp | str,
    config: DownloadConfig | None = None,
) -> pd.DataFrame | None:
    """Download one trading date of TXO option data from TAIFEX.

    Returns ``None`` when the exchange reports no data for the requested date.
    Raises ``TaifexDownloadError`` after repeated network or parsing failures.
    """

    config = config or DownloadConfig()
    date_text = _format_query_date(trading_date)
    payload = {
        "down_type": "1",
        "commodity_id": "TXO",
        "commodity_id2": "",
        "queryStartDate": date_text,
        "queryEndDate": date_text,
    }
    headers = {"User-Agent": USER_AGENT}

    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = requests.post(
                TAIFEX_URL,
                data=payload,
                headers=headers,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            response.encoding = "big5"
            text = response.text.lstrip("\ufeff")

            if not text.strip() or "查無資料" in text:
                return None
            if "<html" in text[:500].lower():
                raise TaifexDownloadError("TAIFEX returned HTML instead of CSV data")

            frame = pd.read_csv(io.StringIO(text), dtype=str)
            frame.columns = [
                str(column).strip().replace('"', "").lstrip("\ufeff")
                for column in frame.columns
            ]
            if frame.empty:
                return None
            return frame
        except (requests.RequestException, pd.errors.ParserError, UnicodeError, TaifexDownloadError) as exc:
            last_error = exc
            if attempt < config.max_attempts:
                time.sleep(config.retry_wait_seconds * attempt)

    raise TaifexDownloadError(
        f"Unable to download {date_text} after {config.max_attempts} attempts: {last_error}"
    )


def infer_expiration_date(contract_code: object) -> pd.Timestamp:
    """Infer settlement date from codes such as 202607, 202607W2, or 202607F1."""

    code = str(contract_code).strip().upper()
    match = re.fullmatch(r"(?P<year>\d{4})(?P<month>\d{2})(?:(?P<kind>[WF])(?P<n>\d+))?", code)
    if not match:
        return pd.NaT

    year = int(match.group("year"))
    month = int(match.group("month"))
    kind = match.group("kind")
    nth = int(match.group("n")) if match.group("n") else 3
    weekday = 4 if kind == "F" else 2  # Friday for F; Wednesday otherwise.

    first_day = pd.Timestamp(year=year, month=month, day=1)
    days_to_weekday = (weekday - first_day.weekday() + 7) % 7
    first_target = first_day + pd.Timedelta(days=days_to_weekday)
    return first_target + pd.Timedelta(weeks=nth - 1)


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    digits = series.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"-": np.nan, "--": np.nan, "": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def preprocess_txo_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize TAIFEX TXO option data and construct a usable quote price."""

    required = {"契約", "交易日期", "到期月份(週別)", "履約價", "買賣權"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing expected TAIFEX columns: {missing}")

    frame = raw.loc[raw["契約"].astype(str).str.strip().eq("TXO")].copy()
    if frame.empty:
        return frame

    mapping = {
        "交易日期": "Date",
        "到期月份(週別)": "Expiration_Month",
        "履約價": "Strike_Price",
        "買賣權": "Option_Type",
        "開盤價": "Open",
        "最高價": "High",
        "最低價": "Low",
        "收盤價": "Close",
        "結算價": "Settlement_Price",
        "最後最佳買價": "Best_Bid",
        "最後最佳賣價": "Best_Ask",
        "交易時段": "Trading_Session",
        "契約到期日": "Expiration_Date",
        "最後交易日": "Expiration_Date",
    }
    frame = frame.rename(columns={key: value for key, value in mapping.items() if key in frame.columns})

    frame["Option_Type"] = (
        frame["Option_Type"].astype(str).str.strip().map({"買權": "Call", "賣權": "Put"})
    )
    frame = frame.loc[frame["Option_Type"].isin(["Call", "Put"])].copy()

    if "Trading_Session" in frame.columns:
        frame["Trading_Session"] = (
            frame["Trading_Session"]
            .astype(str)
            .str.strip()
            .map({"一般": "Regular", "盤後": "AfterHours"})
            .fillna(frame["Trading_Session"])
        )
        frame = frame.loc[frame["Trading_Session"].eq("Regular")].copy()

    for column in [
        "Strike_Price",
        "Open",
        "High",
        "Low",
        "Close",
        "Settlement_Price",
        "Best_Bid",
        "Best_Ask",
    ]:
        if column in frame.columns:
            frame[column] = _to_numeric(frame[column])

    frame["Date"] = _parse_yyyymmdd(frame["Date"])

    if "Expiration_Date" in frame.columns:
        parsed_expiration = _parse_yyyymmdd(frame["Expiration_Date"])
    else:
        parsed_expiration = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    inferred_expiration = frame["Expiration_Month"].apply(infer_expiration_date)
    frame["Expiration_Date"] = parsed_expiration.fillna(inferred_expiration)

    bid = frame.get("Best_Bid", pd.Series(np.nan, index=frame.index))
    ask = frame.get("Best_Ask", pd.Series(np.nan, index=frame.index))
    valid_market = bid.notna() & ask.notna() & bid.ge(0) & ask.ge(bid)
    frame["Mid_Price"] = np.where(valid_market, (bid + ask) / 2.0, np.nan)

    settlement = frame.get("Settlement_Price", pd.Series(np.nan, index=frame.index))
    close = frame.get("Close", pd.Series(np.nan, index=frame.index))
    frame["Price"] = frame["Mid_Price"].fillna(settlement).fillna(close)

    frame = frame.dropna(
        subset=["Date", "Expiration_Date", "Strike_Price", "Option_Type", "Price"]
    )
    frame = frame.loc[(frame["Strike_Price"] > 0) & (frame["Price"] >= 0)].copy()
    return frame


def _delta_k(strikes: np.ndarray) -> np.ndarray:
    if len(strikes) < 3:
        raise ValueError("At least three strikes are required")
    result = np.zeros(len(strikes), dtype=float)
    result[0] = strikes[1] - strikes[0]
    result[-1] = strikes[-1] - strikes[-2]
    result[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    return result


def calculate_term_structure(
    frame: pd.DataFrame,
    as_of: date | datetime | pd.Timestamp | None = None,
    risk_free_rate: float = 0.01,
) -> pd.DataFrame:
    """Calculate one VIX-style implied-volatility estimate per expiration."""

    if frame.empty:
        return pd.DataFrame(
            columns=["Date", "Expiration_Month", "Expiration_Date", "Days_to_Exp", "Forward", "K0", "VIX"]
        )

    valuation_date = pd.Timestamp(as_of or frame["Date"].dropna().max()).normalize()
    working = frame.copy()
    working["T"] = (working["Expiration_Date"] - valuation_date).dt.total_seconds() / (365.0 * 24 * 3600)
    working = working.loc[working["T"] > 0].copy()

    records: list[dict[str, object]] = []
    for expiration_month, group in working.groupby("Expiration_Month", sort=False):
        expiration_date = group["Expiration_Date"].dropna().iloc[0]
        t = float(group["T"].dropna().iloc[0])
        if not math.isfinite(t) or t <= 0:
            continue

        pivot = group.pivot_table(
            index="Strike_Price",
            columns="Option_Type",
            values="Price",
            aggfunc="median",
        ).sort_index()

        if not {"Call", "Put"}.issubset(pivot.columns):
            continue
        paired = pivot.dropna(subset=["Call", "Put"], how="any")
        if len(paired) < 1:
            continue

        difference = (paired["Call"] - paired["Put"]).abs()
        forward_strike = float(difference.idxmin())
        call_price = float(paired.loc[forward_strike, "Call"])
        put_price = float(paired.loc[forward_strike, "Put"])
        forward = forward_strike + math.exp(risk_free_rate * t) * (call_price - put_price)

        eligible_k0 = pivot.index[pivot.index <= forward]
        if len(eligible_k0) == 0:
            continue
        k0 = float(eligible_k0.max())

        q_values: dict[float, float] = {}
        for strike, row in pivot.iterrows():
            strike_value = float(strike)
            if strike_value < k0 and pd.notna(row.get("Put")):
                q_values[strike_value] = float(row["Put"])
            elif strike_value > k0 and pd.notna(row.get("Call")):
                q_values[strike_value] = float(row["Call"])
            elif strike_value == k0 and pd.notna(row.get("Call")) and pd.notna(row.get("Put")):
                q_values[strike_value] = float((row["Call"] + row["Put"]) / 2.0)

        q = pd.Series(q_values, name="Q").replace([np.inf, -np.inf], np.nan).dropna()
        q = q.loc[q >= 0].sort_index()
        if len(q) < 3:
            continue

        strikes = q.index.to_numpy(dtype=float)
        contribution = (
            _delta_k(strikes)
            / np.square(strikes)
            * math.exp(risk_free_rate * t)
            * q.to_numpy(dtype=float)
        )
        sigma_squared = (
            (2.0 / t) * contribution.sum()
            - (1.0 / t) * ((forward / k0) - 1.0) ** 2
        )
        if not math.isfinite(sigma_squared) or sigma_squared <= 0:
            continue

        records.append(
            {
                "Date": valuation_date.strftime("%Y-%m-%d"),
                "Expiration_Month": str(expiration_month),
                "Expiration_Date": pd.Timestamp(expiration_date).strftime("%Y-%m-%d"),
                "Days_to_Exp": round(t * 365.0, 4),
                "Forward": round(forward, 4),
                "K0": round(k0, 4),
                "VIX": round(100.0 * math.sqrt(sigma_squared), 4),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("Days_to_Exp").reset_index(drop=True)


def find_latest_available_term_structure(
    candidate_dates: Iterable[date | datetime | pd.Timestamp],
    risk_free_rate: float = 0.01,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Try candidate dates in order and return the first usable term structure."""

    errors: list[str] = []
    for candidate in candidate_dates:
        candidate_timestamp = pd.Timestamp(candidate).normalize()
        try:
            raw = get_taifex_options_data(candidate_timestamp)
            if raw is None or raw.empty:
                continue
            cleaned = preprocess_txo_data(raw)
            result = calculate_term_structure(
                cleaned,
                as_of=candidate_timestamp,
                risk_free_rate=risk_free_rate,
            )
            if not result.empty:
                return candidate_timestamp, result
        except (TaifexDownloadError, ValueError, KeyError) as exc:
            errors.append(f"{candidate_timestamp.date()}: {exc}")

    detail = "; ".join(errors[-3:]) if errors else "No exchange data was available"
    raise TaifexDownloadError(f"No usable TXO term structure found. {detail}")
