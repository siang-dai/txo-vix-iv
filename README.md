# TXO VIX-style Term Structure

A reproducible Python project that downloads end-of-day Taiwan Futures Exchange TXO option data, estimates a VIX-style implied variance for each expiration, and generates a daily static term-structure chart.

> **Important:** This is a research estimate, not the official TAIWAN VIX. The official exchange index uses additional quote filters, exact time-to-expiration conventions, and interpolation to a constant 30-day maturity.

## Repository structure

```text
src/txo_vix.py             downloader, cleaning, expiration inference, calculation
src/update_latest.py       latest-date search, history update, static chart
src/backfill.py            resumable historical backfill for local use
data/                      generated history (created on first successful run)
output/                    latest PNG, SVG, and JSON metadata
.github/workflows/update.yml  weekday GitHub Actions workflow
tests/                     basic unit tests
```

## Run locally on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python src/update_latest.py

# Optional historical backfill; run locally and keep the polite delay.
python src/backfill.py --start 2020-01-01 --end 2026-08-03 --sleep 2
```

Generated files:

- `output/latest_term_structure.png`
- `output/latest_term_structure.svg`
- `output/latest.json`
- `data/vix_term_structure_history.csv`

## Daily automation

The workflow runs at 19:27 Asia/Taipei on weekdays (11:27 UTC) and can also be started manually from the repository's Actions tab.

In GitHub, confirm:

1. Settings → Actions → General → Workflow permissions → **Read and write permissions**.
2. The workflow file is on the default `main` branch.
3. Run the workflow manually once to verify that TAIFEX accepts the GitHub-hosted runner's request.

## Embed the latest chart

Replace `siang-dai`:

```html
<img src="https://raw.githubusercontent.com/siang-dai/txo-vix-term-structure/main/output/latest_term_structure.png" alt="Latest TXO VIX-style term structure">
```

## Methodology limitations and next improvements

- The current estimator uses daily dates rather than minute-precise expiration time.
- It does not reproduce every TAIFEX quote-quality and sequence filter.
- The risk-free rate is a fixed parameter by default.
- It reports one estimate per expiration; it does not interpolate to a constant 30-day index.
- GitHub-hosted runners use shared IP addresses. TAIFEX may occasionally throttle or reject automated requests.

For an academic paper, document every deviation from the official methodology and validate the result against official published TAIWAN VIX data.
