"""Black-76 implied-volatility calibration for TXO using same-expiry TX/MTX futures.

The implementation is adapted from the user's research script.  It intentionally:
- uses regular-session end-of-day data;
- uses option settlement prices by default;
- accepts only futures whose expiration date exactly equals the option expiration;
- performs no nearest-expiry substitution, interpolation, extrapolation, or IV smoothing.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

RISK_FREE_RATE = 0.01
MAX_IV = 5.0
MAX_EXPIRATIONS = 4
USE_OPTION_MID = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

PRODUCT_LABELS = {
    "TX": "TAIEX Futures (TX)",
    "MTX": "Mini-TAIEX Futures (MTX)",
}


def read_csv_text(text: str) -> pd.DataFrame:
    lines = [line.rstrip("\r\n,") for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()
    frame = pd.read_csv(io.StringIO("\n".join(lines)), dtype=str)
    frame.columns = [
        str(column).replace("\ufeff", "").replace('"', "").strip()
        for column in frame.columns
    ]
    return frame


def post_csv(url: str, payload: dict[str, str]) -> pd.DataFrame:
    response = requests.post(url, data=payload, headers=HEADERS, timeout=30)
    response.raise_for_status()
    text = response.content.decode("big5", errors="replace")
    if not text.strip() or "查無資料" in text:
        return pd.DataFrame()
    if "<html" in text[:500].lower():
        raise RuntimeError("TAIFEX returned HTML instead of CSV data")
    return read_csv_text(text)


def num(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "nan", "None"}:
        return np.nan
    return float(pd.to_numeric(text, errors="coerce"))


def parse_date(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    offset = (weekday - first.weekday() + 7) % 7
    return first + pd.Timedelta(days=offset, weeks=nth - 1)


def contract_expiry(code: object) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{4})(\d{2})(?:([WF])(\d+))?", str(code).strip().upper())
    if not match:
        return pd.NaT

    year, month = int(match.group(1)), int(match.group(2))
    kind, nth_text = match.group(3), match.group(4)
    if kind is None:
        return nth_weekday(year, month, weekday=2, nth=3)
    return nth_weekday(
        year,
        month,
        weekday=2 if kind == "W" else 4,
        nth=int(nth_text),
    )


def get_options(date: pd.Timestamp | str) -> pd.DataFrame:
    date_text = pd.Timestamp(date).strftime("%Y/%m/%d")
    raw = post_csv(
        "https://www.taifex.com.tw/cht/3/optDataDown",
        {
            "down_type": "1",
            "commodity_id": "TXO",
            "commodity_id2": "",
            "queryStartDate": date_text,
            "queryEndDate": date_text,
        },
    )
    if raw.empty:
        return raw

    required = {"契約", "交易日期", "到期月份(週別)", "履約價", "買賣權"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing expected option columns: {missing}")

    frame = raw[raw["契約"].astype(str).str.strip().eq("TXO")].copy()
    frame = frame.rename(
        columns={
            "交易日期": "Date",
            "到期月份(週別)": "Contract",
            "履約價": "Strike",
            "買賣權": "Type",
            "結算價": "Settlement",
            "最後最佳買價": "Bid",
            "最後最佳賣價": "Ask",
            "交易時段": "Session",
            "契約到期日": "Expiry",
            "最後交易日": "Last_Trading_Date",
        }
    )

    if "Session" in frame.columns:
        frame = frame[frame["Session"].astype(str).str.strip().eq("一般")].copy()

    frame["Type"] = frame["Type"].astype(str).str.strip().map({"買權": "C", "賣權": "P"})
    frame["Strike"] = frame["Strike"].map(num)
    for column in ["Settlement", "Bid", "Ask"]:
        frame[column] = frame[column].map(num) if column in frame.columns else np.nan

    valid_mid = frame["Bid"].gt(0) & frame["Ask"].gt(0) & frame["Ask"].ge(frame["Bid"])
    if USE_OPTION_MID:
        frame["Price"] = np.where(
            valid_mid,
            (frame["Bid"] + frame["Ask"]) / 2.0,
            frame["Settlement"],
        )
        frame["Price_Source"] = np.where(valid_mid, "mid", "settlement")
    else:
        frame["Price"] = frame["Settlement"]
        frame["Price_Source"] = "settlement"

    frame["Date"] = parse_date(frame["Date"]).dt.normalize()
    frame["Contract"] = frame["Contract"].astype(str).str.strip().str.upper()
    frame["Expiry"] = parse_date(frame["Expiry"]) if "Expiry" in frame.columns else pd.NaT
    if "Last_Trading_Date" in frame.columns:
        frame["Expiry"] = frame["Expiry"].fillna(parse_date(frame["Last_Trading_Date"]))
    frame["Expiry"] = frame["Expiry"].fillna(frame["Contract"].map(contract_expiry))

    columns = [
        "Date",
        "Contract",
        "Expiry",
        "Strike",
        "Type",
        "Price",
        "Price_Source",
        "Settlement",
        "Bid",
        "Ask",
    ]
    return frame[columns].dropna(subset=["Date", "Expiry", "Strike", "Type", "Price"])


def get_futures(date: pd.Timestamp | str) -> pd.DataFrame:
    """Download TX and MTX separately; do not let TX overwrite MTX."""

    date_text = pd.Timestamp(date).strftime("%Y/%m/%d")
    raw = post_csv(
        "https://www.taifex.com.tw/cht/3/futDataDown",
        {
            "down_type": "1",
            "commodity_id": "all",
            "commodity_id2": "",
            "queryStartDate": date_text,
            "queryEndDate": date_text,
        },
    )
    if raw.empty:
        return raw

    required = {"交易時段", "契約", "到期月份(週別)"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing expected futures columns: {missing}")

    frame = raw[
        raw["交易時段"].astype(str).str.strip().eq("一般")
        & raw["契約"].astype(str).str.strip().isin(["TX", "MTX"])
    ].copy()
    frame["Product"] = frame["契約"].astype(str).str.strip()
    frame["Contract"] = frame["到期月份(週別)"].astype(str).str.strip().str.upper()
    frame = frame[frame["Contract"].str.match(r"^\d{6}(?:[WF]\d+)?$", na=False)].copy()

    settlement = (
        frame["結算價"].map(num)
        if "結算價" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    last_price = (
        frame["最後成交價"].map(num)
        if "最後成交價" in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    frame["Price"] = settlement.fillna(last_price)
    frame = frame[frame["Price"].gt(0)].copy()

    expiry_column = next(
        (column for column in ["契約到期日", "最後交易日", "到期日"] if column in frame.columns),
        None,
    )
    frame["Expiry"] = parse_date(frame[expiry_column]) if expiry_column else pd.NaT
    frame["Expiry"] = frame["Expiry"].fillna(frame["Contract"].map(contract_expiry))

    # De-duplicate only within the same product and contract.
    frame = (
        frame.sort_values(["Product", "Expiry", "Contract"])
        .drop_duplicates(["Product", "Contract"], keep="last")
    )
    return frame[["Product", "Contract", "Expiry", "Price"]].dropna(subset=["Expiry", "Price"])


def select_same_expiry_futures_price(
    option_expiry: pd.Timestamp,
    futures: pd.DataFrame,
    product: str,
) -> tuple[float, str]:
    """Return a futures price only when product and expiration both match exactly."""

    curve = futures[futures["Product"].eq(product)].dropna(subset=["Expiry", "Price"]).copy()
    if curve.empty:
        raise ValueError(f"No usable {product} futures prices")

    normalized_expiry = pd.Timestamp(option_expiry).normalize()
    curve["Expiry"] = pd.to_datetime(curve["Expiry"]).dt.normalize()
    matched = curve[curve["Expiry"].eq(normalized_expiry)].copy()
    if matched.empty:
        raise ValueError(
            f"No {product} futures contract with expiry {normalized_expiry.date()}"
        )

    row = matched.sort_values("Contract").iloc[0]
    return float(row["Price"]), str(row["Contract"])


def black76(
    forward: float,
    strike: float,
    maturity: float,
    sigma: float,
    option_type: str,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    discount = math.exp(-risk_free_rate * maturity)
    if sigma <= 0:
        intrinsic = (
            max(forward - strike, 0.0)
            if option_type == "C"
            else max(strike - forward, 0.0)
        )
        return discount * intrinsic

    volatility_time = sigma * math.sqrt(maturity)
    d1 = (
        math.log(forward / strike) + 0.5 * sigma * sigma * maturity
    ) / volatility_time
    d2 = d1 - volatility_time
    if option_type == "C":
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    option_type: str,
    risk_free_rate: float = RISK_FREE_RATE,
    max_iv: float = MAX_IV,
) -> float:
    if not all(np.isfinite([price, forward, strike, maturity])):
        return np.nan
    if min(price, forward, strike, maturity) <= 0:
        return np.nan

    discount = math.exp(-risk_free_rate * maturity)
    intrinsic = discount * (
        max(forward - strike, 0.0)
        if option_type == "C"
        else max(strike - forward, 0.0)
    )
    upper = discount * (forward if option_type == "C" else strike)
    if price <= intrinsic + 1e-10 or price >= upper:
        return np.nan

    def objective(sigma: float) -> float:
        return black76(
            forward,
            strike,
            maturity,
            sigma,
            option_type,
            risk_free_rate,
        ) - price

    if objective(max_iv) < 0:
        return np.nan
    try:
        return float(brentq(objective, 1e-8, max_iv, maxiter=200))
    except ValueError:
        return np.nan


def calibrate_product(
    options: pd.DataFrame,
    futures: pd.DataFrame,
    trade_date: pd.Timestamp,
    product: str,
    risk_free_rate: float = RISK_FREE_RATE,
    max_expirations: int = MAX_EXPIRATIONS,
) -> pd.DataFrame:
    """Invert Black-76 for OTM TXO quotes using exact-expiry TX or MTX futures."""

    if product not in PRODUCT_LABELS:
        raise ValueError(f"Unsupported futures product: {product}")

    trade_date = pd.Timestamp(trade_date).normalize()
    working = options.copy()
    working["Days_To_Expiry"] = (working["Expiry"] - trade_date).dt.days
    working["T"] = working["Days_To_Expiry"] / 365.0
    working = working[
        working["T"].gt(0)
        & working["Strike"].gt(0)
        & working["Price"].gt(0)
    ].copy()

    product_futures = futures[futures["Product"].eq(product)].copy()
    futures_expiries = set(pd.to_datetime(product_futures["Expiry"]).dt.normalize())
    expirations = (
        working[["Contract", "Expiry"]]
        .drop_duplicates()
        .assign(Expiry=lambda frame: pd.to_datetime(frame["Expiry"]).dt.normalize())
    )
    expirations = (
        expirations[expirations["Expiry"].isin(futures_expiries)]
        .sort_values(["Expiry", "Contract"])
        .head(max_expirations)
    )

    results: list[pd.DataFrame] = []
    for item in expirations.itertuples(index=False):
        chain = working[working["Contract"].eq(item.Contract)].copy()
        try:
            forward, futures_contract = select_same_expiry_futures_price(
                item.Expiry,
                futures,
                product,
            )
        except ValueError:
            continue

        chain["Futures_Product"] = product
        chain["Forward"] = forward
        chain["Futures_Contract"] = futures_contract
        chain["Forward_Source"] = f"same expiry {product} {futures_contract}"
        chain["IV"] = chain.apply(
            lambda row: implied_vol(
                row["Price"],
                forward,
                row["Strike"],
                row["T"],
                row["Type"],
                risk_free_rate,
            ),
            axis=1,
        )
        chain["OTM"] = (
            ((chain["Type"].eq("C")) & chain["Strike"].ge(forward))
            | ((chain["Type"].eq("P")) & chain["Strike"].lt(forward))
        )
        chain["Plot_Eligible"] = chain["OTM"] & chain["IV"].notna()
        results.append(chain)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def build_iv_payload(
    calibrated_products: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    risk_free_rate: float = RISK_FREE_RATE,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Serialize only the original eligible scatter points, without smoothing.

    Row arrays keep the public payload small. IV is converted from a decimal to
    percentage points for display; full-precision calculations remain in CSV.
    """
    trade_date = pd.Timestamp(trade_date).normalize()
    products = {}
    for product in PRODUCT_LABELS:
        calibrated = calibrated_products[product]
        curves = []
        if not calibrated.empty:
            if not pd.to_datetime(calibrated["Date"]).dt.normalize().eq(trade_date).all():
                raise ValueError("IV observations must match the publication date")
            if not calibrated["Futures_Product"].eq(product).all():
                raise ValueError("IV observations must match the futures product")
            expirations = (calibrated[["Contract", "Expiry"]].drop_duplicates()
                           .sort_values(["Expiry", "Contract"]))
            for item in expirations.itertuples(index=False):
                chain = calibrated[calibrated["Contract"].eq(item.Contract)]
                eligible = chain.loc[chain["Plot_Eligible"]].sort_values("Strike")
                forward = float(chain["Forward"].iloc[0])
                values = eligible[["Strike", "IV", "Price"]].to_numpy(dtype=float)
                if (not np.isfinite(forward) or forward <= 0
                        or not np.isfinite(values).all() or (values <= 0).any()):
                    raise ValueError("Eligible IV points must be finite and positive")
                if not eligible["Type"].isin(["P", "C"]).all():
                    raise ValueError("Unknown option type in IV points")
                curve = {
                    "option_contract": str(item.Contract),
                    "expiry": pd.Timestamp(item.Expiry).strftime("%Y-%m-%d"),
                    "days_to_expiry": int(chain["Days_To_Expiry"].iloc[0]),
                    "futures_contract": str(chain["Futures_Contract"].iloc[0]),
                    "forward": round(forward, 4),
                }
                for kind, key in (("P", "puts"), ("C", "calls")):
                    points = eligible.loc[eligible["Type"].eq(kind), ["Strike", "IV", "Price"]]
                    curve[key] = [[round(float(k), 4), round(float(iv) * 100, 6), round(float(price), 6)]
                                  for k, iv, price in points.itertuples(index=False, name=None)]
                curves.append(curve)
        products[product] = {
            "matched_expirations": len(curves),
            "plotted_points": sum(len(c["puts"]) + len(c["calls"]) for c in curves),
            "data": f"latest_{product.lower()}_iv_points.csv",
            "curves": curves,
        }
    return {
        "schema": 1,
        "as_of_date": trade_date.strftime("%Y-%m-%d"),
        "generated_at_taipei": generated_at or pd.Timestamp.now(tz="Asia/Taipei").isoformat(timespec="seconds"),
        "timezone": "Asia/Taipei",
        "method": "Black-76 IV inversion using exact same-expiry futures",
        "option_price_source": "settlement" if not USE_OPTION_MID else "mid_then_settlement",
        "risk_free_rate": risk_free_rate,
        "point_columns": ["strike", "iv_pct", "option_price"],
        "products": products,
    }


def update_iv_outputs(
    trade_date: pd.Timestamp,
    output_dir: Path,
    data_dir: Path,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, dict[str, object]]:
    """Publish compact TX/MTX IV chart data and full-precision research CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    options = get_options(trade_date)
    futures = get_futures(trade_date)
    if options.empty or futures.empty:
        raise RuntimeError("Option or futures data is empty for the selected trading date")

    calibrated_products = {
        product: calibrate_product(options, futures, trade_date, product, risk_free_rate=risk_free_rate)
        for product in PRODUCT_LABELS
    }
    payload = build_iv_payload(calibrated_products, trade_date, risk_free_rate)
    # Validate the whole publication before replacing the previous chart data.
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    for product, calibrated in calibrated_products.items():
        calibrated.to_csv(data_dir / f"latest_{product.lower()}_iv_points.csv", index=False, encoding="utf-8-sig")
    (output_dir / "latest_iv.json").write_text(encoded, encoding="utf-8")
    return {product: {key: value for key, value in details.items() if key != "curves"}
            for product, details in payload["products"].items()}
