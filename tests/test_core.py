import pandas as pd

from src.option_volume import calculate_expiration_volume
from src.txo_vix import (
    calculate_term_structure,
    infer_expiration_date,
    preprocess_txo_data,
    replace_trading_day,
)


def test_quote_filter_drops_wide_quotes_before_fallback():
    raw = pd.DataFrame({
        "契約": "TXO", "交易日期": "20260819", "到期月份(週別)": "202609",
        "履約價": range(23000, 23009), "買賣權": ["賣權", "買權", "買權"] * 3,
        "最後最佳買價": [0.2, 0.6, 1, 10, 0, 0, 100, "-", 1],
        "最後最佳賣價": [4520, 565, 51, 100, 50, 51, 160, 80, 2],
        "結算價": 9, "收盤價": 8,
    })
    cleaned = preprocess_txo_data(raw)
    assert cleaned.set_index("Strike_Price")["Price"].to_dict() == {
        23002: 26, 23003: 55, 23004: 25, 23006: 130, 23007: 9, 23008: 1.5,
    }
    # Only one eligible strike remains for this slice.
    assert calculate_term_structure(cleaned.iloc[:1]).empty


def test_recalculation_replaces_all_expirations_for_the_day():
    history = pd.DataFrame({
        "Date": ["2026-08-18", "2026-08-19", "2026-08-19"],
        "Days_to_Exp": [29, 28, 63],
        "VIX": [31, 40, 34],
    })
    result = history.iloc[[1]].assign(VIX=32)
    updated = replace_trading_day(history, "2026-08-19", result)
    assert updated["VIX"].tolist() == [31, 32]
    assert replace_trading_day(updated, "2026-08-19", result.iloc[:0])["VIX"].tolist() == [31]


def test_infer_monthly_expiration_is_third_wednesday():
    assert infer_expiration_date("202607") == pd.Timestamp("2026-07-15")


def test_infer_weekly_expirations():
    assert infer_expiration_date("202607W2") == pd.Timestamp("2026-07-08")
    assert infer_expiration_date("202607F1") == pd.Timestamp("2026-07-03")


def test_calculate_term_structure_returns_positive_value():
    rows = []
    for strike, call, put in [
        (90, 12.0, 1.0),
        (95, 8.0, 2.0),
        (100, 5.0, 4.8),
        (105, 2.4, 7.0),
        (110, 1.1, 11.0),
    ]:
        rows.extend(
            [
                {
                    "Date": pd.Timestamp("2026-07-01"),
                    "Expiration_Month": "202608",
                    "Expiration_Date": pd.Timestamp("2026-08-19"),
                    "Strike_Price": strike,
                    "Option_Type": "Call",
                    "Price": call,
                },
                {
                    "Date": pd.Timestamp("2026-07-01"),
                    "Expiration_Month": "202608",
                    "Expiration_Date": pd.Timestamp("2026-08-19"),
                    "Strike_Price": strike,
                    "Option_Type": "Put",
                    "Price": put,
                },
            ]
        )

    result = calculate_term_structure(pd.DataFrame(rows), as_of="2026-07-01")
    assert len(result) == 1
    assert result.loc[0, "VIX"] > 0


def test_expiry_day_volume_and_post_settlement_open_interest():
    rows = pd.DataFrame({
        "契約": ["TXO"] * 5, "交易日期": ["2026/08/19"] * 5,
        "到期月份(週別)": ["202608", "202609", "202609", "202609", "202609"],
        "契約到期日": ["20260819"] + ["20260916"] * 4,
        "履約價": [30000, 30450, 30000, 31500, 31500],
        "買賣權": ["賣權", "買權", "買權", "賣權", "賣權"],
        "交易時段": ["一般", "一般", "盤後", "一般", "盤後"],
        "成交量": [7, 8, 2, 11, 5], "未沖銷契約數": ["99", "40", "-", "20", "-"],
    })
    result = calculate_expiration_volume(rows, "2026-08-19")
    counts = ["Call_Volume", "Put_Volume", "Volume", "Call_OI", "Put_OI"]
    assert result[counts].values.tolist() == [[0, 7, 7, 0, 0], [10, 16, 26, 40, 20]]
    assert result[["First_Strike_Bin", "Call_Bins_500", "Put_Bins_500"]].values.tolist() == [
        [30000, "0", "7"], [30000, "10 0 0 0", "0 0 0 16"],
    ]
