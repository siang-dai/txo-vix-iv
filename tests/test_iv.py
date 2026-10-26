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


def calibrated_fixture():
    from src.txo_iv import calibrate_product
    date, expiry = pd.Timestamp('2026-08-03'), pd.Timestamp('2026-08-19')
    options = pd.DataFrame({
        'Date': [date] * 5, 'Contract': ['202608'] * 5, 'Expiry': [expiry] * 5,
        'Strike': [19800., 20200., 20200., 19800., 21000.],
        'Type': ['P', 'C', 'P', 'C', 'C'],
        'Price': [black76(20000, k, 16 / 365, .22, t) for k, t in
                  [(19800, 'P'), (20200, 'C'), (20200, 'P'), (19800, 'C')]] + [1e9],
        'Price_Source': ['settlement'] * 5,
    })
    futures = pd.DataFrame({'Product': ['TX', 'MTX'], 'Contract': ['202608'] * 2,
                            'Expiry': [expiry] * 2, 'Price': [20000., 19990.]})
    calibrated = {product: calibrate_product(options, futures, date, product)
                  for product in ('TX', 'MTX')}
    return date, options, futures, calibrated


def test_chart_payload_preserves_eligible_points_and_converts_iv_to_percent():
    import json
    from src.txo_iv import build_iv_payload
    date, _, _, calibrated = calibrated_fixture()
    payload = build_iv_payload(calibrated, date)
    json.dumps(payload, allow_nan=False)
    for product, expected_forward in [('TX', 20000.), ('MTX', 19990.)]:
        result = payload['products'][product]
        curve = result['curves'][0]
        assert curve['forward'] == expected_forward
        assert curve['expiry'] == '2026-08-19'
        assert curve['days_to_expiry'] == 16
        assert result['matched_expirations'] == 1 and result['plotted_points'] == 2
        for kind, key in [('P', 'puts'), ('C', 'calls')]:
            source = calibrated[product].loc[lambda f: f.Plot_Eligible & f.Type.eq(kind)].iloc[0]
            assert curve[key] == [[source.Strike, round(source.IV * 100, 6), round(source.Price, 6)]]
    assert payload['products']['TX']['curves'][0]['puts'][0][1] == 22.


def test_chart_payload_handles_unmatched_products_and_rejects_bad_points():
    import pytest
    from src.txo_iv import build_iv_payload
    date, _, _, calibrated = calibrated_fixture()
    calibrated['MTX'] = pd.DataFrame()
    payload = build_iv_payload(calibrated, date)
    assert payload['products']['MTX']['curves'] == []
    assert payload['products']['MTX']['plotted_points'] == 0
    calibrated['TX'].loc[0, 'IV'] = float('inf')
    with pytest.raises(ValueError, match='finite and positive'):
        build_iv_payload(calibrated, date)


def test_update_publishes_json_and_csv_without_png(monkeypatch, tmp_path):
    import json
    import src.txo_iv as iv
    date, options, futures, _ = calibrated_fixture()
    monkeypatch.setattr(iv, 'get_options', lambda _: options)
    monkeypatch.setattr(iv, 'get_futures', lambda _: futures)
    summary = iv.update_iv_outputs(date, tmp_path / 'output', tmp_path / 'data')
    payload = json.loads((tmp_path / 'output/latest_iv.json').read_text())
    assert payload['products']['TX']['plotted_points'] == summary['TX']['plotted_points'] == 2
    assert len(list((tmp_path / 'data').glob('*.csv'))) == 2
    assert not list(tmp_path.rglob('*.png'))
    # A failed download must preserve the last published chart data.
    original = (tmp_path / 'output/latest_iv.json').read_bytes()
    monkeypatch.setattr(iv, 'get_options', lambda _: pd.DataFrame())
    import pytest
    with pytest.raises(RuntimeError, match='data is empty'):
        iv.update_iv_outputs(date, tmp_path / 'output', tmp_path / 'data')
    assert (tmp_path / 'output/latest_iv.json').read_bytes() == original
