import math

import pandas as pd

from src.txo_iv import black76, implied_vol, select_same_expiry_futures_price


def test_black76_round_trip_call():
    price = black76(20000.0, 20200.0, 30 / 365, 0.22, "C", 0.01)
    recovered = implied_vol(price, 20000.0, 20200.0, 30 / 365, "C", 0.01)
    assert math.isclose(recovered, 0.22, rel_tol=1e-6)


def test_product_and_expiry_must_both_match():
    futures = pd.DataFrame(
        {
            "Product": ["TX", "MTX", "TX"],
            "Contract": ["202608", "202608", "202609"],
            "Expiry": pd.to_datetime(["2026-08-19", "2026-08-19", "2026-09-16"]),
            "Price": [20000.0, 19998.0, 20100.0],
        }
    )
    tx_price, tx_contract = select_same_expiry_futures_price(
        pd.Timestamp("2026-08-19"), futures, "TX"
    )
    mtx_price, mtx_contract = select_same_expiry_futures_price(
        pd.Timestamp("2026-08-19"), futures, "MTX"
    )
    assert (tx_price, tx_contract) == (20000.0, "202608")
    assert (mtx_price, mtx_contract) == (19998.0, "202608")
