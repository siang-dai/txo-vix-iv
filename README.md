# TXO VIX Term Structure and IV Curves

This project downloads Taiwan index option and futures data from the **Taiwan Futures Exchange (TAIFEX)** and constructs:

1. **TXO VIX-style term structure**
   Implied volatility is calculated for each available TXO expiration using a methodology based on the Cboe VIX framework. See [Cboe Volatility Index Mathematics Methodology](https://cdn.cboe.com/resources/indices/Cboe_Volatility_Index_Mathematics_Methodology.pdf).

2. **TXO implied-volatility curves**
   Strike-level implied volatilities are recovered using the [Black-76 formula](https://www.sciencedirect.com/science/article/abs/pii/0304405X76900246). For each TXO expiration, the underlying futures price is taken from a futures contract with the **same expiration date**:

   * **TX** for the TX-based IV curve
   * **MTX** for the MTX-based IV curve

   Only exact expiration-date matches are used; no interpolation, extrapolation, or option-implied forward is applied.

> **Note:** The VIX measure in this repository is a VIX-style term-structure calculation and should not be interpreted as the official TAIWAN VIX published by TAIFEX.

## Latest outputs

The repository publishes the VIX term-structure chart and compact data for the
interactive TX- and MTX-based IV curves on the [TXO Observatory](https://siang-dai.github.io/txo.html).

```text
output/latest_term_structure.png
output/latest_iv.json
```

`latest_iv.json` contains both products, up to four same-expiry contracts each,
and only the eligible OTM scatter points. The browser reuses the page's Plotly
library; it no longer downloads IV PNGs. The daily workflow updates the JSON
alongside the existing full-precision CSV observations, without generating IV images.

## Repository structure

```text
src/txo_vix.py               VIX-style term-structure calculation
src/txo_iv.py                Black-76 implied-volatility calculation
src/update_latest.py         latest-data search and chart generation
src/backfill.py              resumable historical backfill

data/                        historical and latest calculated data
output/                      latest charts and metadata
tests/                       unit tests
.github/workflows/update.yml automated GitHub Actions workflow
```

## Run locally on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m pytest -q
python src/update_latest.py
```

Optional historical backfill:

```bash
python src/backfill.py \
  --start 2020-01-01 \
  --end 2026-08-03 \
  --sleep 2
```

## Generated files

Main chart outputs:

```text
output/latest_term_structure.png
output/latest_term_structure.svg
```

Metadata and interactive chart data:

```text
output/latest.json
output/latest_iv.json
```

Data:

```text
data/vix_term_structure_history.csv
data/latest_tx_iv_points.csv
data/latest_mtx_iv_points.csv
```

## Embed the latest charts

The latest figures can be embedded directly from this repository.

### VIX-style term structure

```html
<img
  src="https://raw.githubusercontent.com/siang-dai/txo-vix-term-structure/main/output/latest_term_structure.png"
  alt="Latest TXO VIX-style term structure">
```

### Interactive TX / MTX IV curves

Fetch `https://raw.githubusercontent.com/siang-dai/txo-vix-iv/main/output/latest_iv.json`.
Schema 1 includes `as_of_date`, `generated_at_taipei`, `option_price_source`,
`risk_free_rate`, and `products.TX` / `products.MTX`.
Each product has `curves`, `matched_expirations`, and `plotted_points`.
Each curve includes the option and futures contract codes, expiry, days to expiry,
matched futures price (`forward`), and separate `puts` / `calls` arrays.
Every point is `[strike, iv_pct, option_price]`, as declared in `point_columns`.

`iv_pct` is the original decimal IV multiplied by 100, rounded to six decimal
places for display. No interpolation or smoothing is introduced. Full-precision
IVs remain in `data/latest_tx_iv_points.csv` and `data/latest_mtx_iv_points.csv`.
Unmatched products have an empty `curves` list; matched expirations without
eligible quotes have empty point lists. They must be displayed as unavailable,
not as zero IV. Both products are validated before the prior JSON is replaced.

## Methodology notes

For the IV curves, option prices are converted into implied volatilities by numerically inverting the Black-76 pricing formula. OTM puts are used below the matched futures price and OTM calls above it.

A TXO expiration is included only when a TX or MTX futures contract with the **same expiration date** is available. As a result, some weekly option expirations may be excluded from the IV plots.

The risk-free rate is currently configurable through the project settings and is set to **1%** by default.
