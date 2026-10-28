# Watt & Wait

Watt & Wait is an end-to-end EV predictive suite for workplace **Post-Charge Dwell Ratio (PCDR)**, driving range/endurance, battery State of Health (SoH), and charging-target recommendations.

PCDR is calculated from session timestamps:

```text
PCDR = (total connection duration - active charging duration) / total connection duration
```

The project includes data validation, leakage-safe historical features, a two-stage training workflow, persisted model bundles, and a Streamlit operations dashboard.

## What is included

```text
.
├── app.py
├── data/
│   ├── generate_mock_ieee_data.py
│   └── ieee_ev_charging.csv
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── nudge_logic.py
│   ├── train_pcdr_model.py
│   └── train_nudge_model.py
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── requirements.txt
```

The checked-in CSV is a 600-row synthetic bootstrap dataset. Regenerate a larger 2,500-row version at any time, or replace it with the locally downloaded IEEE DataPort CSV. Trained `.pkl` artifacts are reproducible and intentionally ignored by Git.

## Quick start

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional: replace the bootstrap CSV with a larger synthetic dataset.
python data\generate_mock_ieee_data.py --rows 2500 --seed 42

python -m src.train_pcdr_model
python -m src.train_nudge_model
python -m src.train_multitarget_models
python -m pytest
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` and use forward slashes in paths.

## Using an IEEE DataPort export

Place the licensed CSV at `data/ieee_ev_charging.csv`, or pass another location to both training commands with `--data PATH`. The canonical required fields are:

- `session_id`, `user_id`, `station_id`
- `connection_start`, `connection_end`
- `charging_start`, `charging_end`
- `energy_kwh`
- optional `user_type` and `department`

The loader converts common names such as `sessionId`, `connectTime`, `disconnectTime`, `chargeStart`, `chargeEnd`, `kwh`, and `energyDeliveredKwh`. Timestamps must be parseable by pandas. Invalid time order, non-positive duration, and negative-energy rows are removed.

## Pipeline design

Stage 1 predicts continuous PCDR with a random-forest regression pipeline. A chronological 80/20 split is used instead of a random split. Features include arrival hour/day, weekend and 07:00–10:00 peak flags, energy, observed average charge rate, categorical station/driver context, and the user's prior PCDR average. The user average is shifted before expanding, so a row never sees its own target. The script reports RMSE, MAE, R², and aggregated feature importance, then writes:

- `models/pcdr_regressor.pkl`
- `models/pcdr_feature_importance.csv`

Stage 2 trains a risk classifier and a nudge-responsiveness classifier, then writes `models/nudge_classifier.pkl`. The product policy remains deterministic and auditable:

- Low risk: predicted PCDR `< 0.15`
- Medium risk: `0.15 <= PCDR < 0.40`
- High risk: predicted PCDR `>= 0.40`
- Nudge priority: `0.60 × predicted PCDR + 0.40 × live congestion`
- Automated-alert recommendation: High risk and occupancy strictly above 70%

The base source schema has no delivered/opened/acted-on notification labels. Therefore, the included responsiveness classifier learns a clearly marked **behavioral proxy**: whether a session's PCDR was at least 20% below the driver's prior average, or below 0.15. This is useful for prototyping only. Replace `response_proxy` in `src/train_nudge_model.py` with a real nudge outcome before production use.

## Dashboard

The Streamlit interface provides four live tabs:

- User, station, arrival, energy, occupancy, user-type, and department inputs
- Predicted PCDR, estimated idle hours, risk, priority, and responsiveness
- A real-time high-risk/high-congestion alert recommendation and reminder time
- Global feature importance, fleet risk distribution, and station comparisons
- A model card with holdout metrics and limitations
- Range and endurance prediction from SOC, temperature, capacity, and driving mode
- SoH and annual degradation estimates from cycle, fast-charge, and thermal inputs
- An auditable 80% vs 100% SOC strategy balancing range, dwell pressure, and health

## Multi-target data and models

`data/generate_mock_ieee_data.py` now creates interconnected `ieee_workplace_charging.csv`, `ieee_vehicle_telemetry.csv`, and `ieee_battery_health.csv` files, plus the legacy charging CSV for backward compatibility. `src.train_multitarget_models` reports RMSE, MAE, and R² for both new regressors and writes `models/model_range.pkl` and `models/model_degradation.pkl`.

The dashboard does not send SMS or app notifications; it produces the decision that an authorized messaging integration can consume.

## Testing and retraining

Run unit tests with `python -m pytest`. Retrain the Stage 1 model before Stage 2 whenever the dataset changes. Both scripts accept `--help` and print progress plus evaluation metrics.

For a production rollout, add schema contracts, dataset/version tracking, group-aware or time-window cross-validation, drift monitoring, occupancy telemetry, real notification outcomes, calibration/fairness review, and an authenticated messaging service. Do not use department or user-derived behavior for consequential employee evaluation.
