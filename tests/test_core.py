import pandas as pd

from src.txo_vix import calculate_term_structure, infer_expiration_date


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
