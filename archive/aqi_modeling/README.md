# AQI Modeling Toolkit

This mini-module ingests the two AQI assets you shared (Noida 2024 daily spreadsheet and CPCB's Delhi PDF bulletin), trains a forecasting model, compares the predictions with the official 1-Jan-2025 reading, and enriches the output with real-world PM2.5/PM10 snapshots from OpenAQ.

## Workflow

1. **Reshape & clean** `AQI_daily_city_level_noida_2024_noida_2024.xlsx` into a tidy daily time series.
2. **Engineer features**: autoregressive lags, rolling means, seasonal (month/day-of-week), and harmonic encodings.
3. **Train** a `HistGradientBoostingRegressor` with time-series cross-validation to avoid leakage.
4. **Forecast** the next _N_ days (default 7). Predictions are stored under `artifacts/aqi/forecast.csv`.
5. **Parse** `AQ-NCR-01012025.pdf` to pull the CPCB snapshot for Delhi + Noida and benchmark our forecast for 1-Jan-2025.
6. **Fetch** the latest PM2.5/PM10 bulletin for Noida via the OpenAQ REST API to ground forecasts in up-to-the-minute measurements/news.

## Quick start

```bash
cd /Users/aayushsharma/Desktop/Overhaul/OVERHAUL-main
source .venv/bin/activate
python -m pip install -r requirements.txt  # already done earlier, rerun if needed
python aqi_modeling/run_aqi_model.py \
  --xlsx AQI_daily_city_level_noida_2024_noida_2024.xlsx \
  --pdf AQ-NCR-01012025.pdf \
  --output-dir artifacts/aqi \
  --forecast-horizon 10 \
  --fetch-openaq \
  --openaq-token YOUR_API_TOKEN \  # optional, improves OpenAQ success rate
  --waqi-token YOUR_WAQI_TOKEN     # optional; defaults to WAQI's public 'demo'
```

Outputs:

- `artifacts/aqi/metrics.json` – CV metrics, CPCB snapshot, and fresh context. If OpenAQ blocks anonymous calls the script falls back to WAQI (demo token) so you still get live PM trends.
- `artifacts/aqi/cv_predictions.csv` – out-of-fold predictions (for diagnostics).
- `artifacts/aqi/forecast.csv` – forward-looking AQI forecasts (if the horizon produces rows).

## Customisation ideas

- Switch `--city` to fetch Delhi/Ghaziabad/Gurugram readings from OpenAQ.
- Extend `MONTH_MAP` + parser if you receive new XLSX schemas.
- Wire the script into a scheduled GitHub Action (e.g., nightly) to keep the `artifacts/aqi` folder fresh.
- Feed the forecasts back into your agents (e.g., `impact_estimator`) for scenario planning.
