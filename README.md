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

The repository automatically generates three main charts:

* TXO VIX-style term structure
* TX-based TXO IV curves
* MTX-based TXO IV curves

The latest figures are stored in:

```text
output/latest_term_structure.png
output/latest_tx_iv_calibration.png
output/latest_mtx_iv_calibration.png
```

Supporting metadata and IV observations are stored in `output/` and `data/`.

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
output/latest_tx_iv_calibration.png
output/latest_mtx_iv_calibration.png
```

Metadata:

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

### TX-based IV curves

```html
<img
  src="https://raw.githubusercontent.com/siang-dai/txo-vix-term-structure/main/output/latest_tx_iv_calibration.png"
  alt="Latest TX-based TXO implied-volatility curves">
```

### MTX-based IV curves

```html
<img
  src="https://raw.githubusercontent.com/siang-dai/txo-vix-term-structure/main/output/latest_mtx_iv_calibration.png"
  alt="Latest MTX-based TXO implied-volatility curves">
```

## Methodology notes

For the IV curves, option prices are converted into implied volatilities by numerically inverting the Black-76 pricing formula. OTM puts are used below the matched futures price and OTM calls above it.

A TXO expiration is included only when a TX or MTX futures contract with the **same expiration date** is available. As a result, some weekly option expirations may be excluded from the IV plots.

The risk-free rate is currently configurable through the project settings and is set to **1%** by default.
