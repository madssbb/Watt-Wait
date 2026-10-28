"""Train range and battery-health regressors from the linked synthetic schemas."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import DEFAULT_WORKPLACE_PATH, load_multitarget_data
from src.feature_engineering import RANGE_FEATURES, HEALTH_FEATURES, add_multitarget_features

ROOT = Path(__file__).resolve().parents[1]
RANGE_MODEL_PATH = ROOT / "models" / "model_range.pkl"
DEGRADATION_MODEL_PATH = ROOT / "models" / "model_degradation.pkl"


def _fit(frame, features, target, path):
    split = int(len(frame) * .8); train, test = frame.iloc[:split], frame.iloc[split:]
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(train[features], train[target]); pred = model.predict(test[features])
    metrics = {"rmse": float(mean_squared_error(test[target], pred)**.5), "mae": float(mean_absolute_error(test[target], pred)), "r2": float(r2_score(test[target], pred)), "train_rows": len(train), "test_rows": len(test)}
    bundle = {"pipeline": model, "features": features, "target": target, "metrics": metrics}
    path.parent.mkdir(exist_ok=True); joblib.dump(bundle, path); print(f"{path.name}: {json.dumps(metrics)}")
    return bundle


def train_models():
    """Train and persist separate range/endurance and SoH/degradation models."""
    if not DEFAULT_WORKPLACE_PATH.exists():
        from data.generate_mock_ieee_data import DATA_DIR, generate_all, generate_mock_data
        # Training callers normally generate explicitly; this guard gives a fresh clone a safe fallback.
        generated = generate_all()
        generated["workplace"].to_csv(DATA_DIR / "ieee_workplace_charging.csv", index=False)
        generated["telemetry"].to_csv(DATA_DIR / "ieee_vehicle_telemetry.csv", index=False)
        generated["health"].to_csv(DATA_DIR / "ieee_battery_health.csv", index=False)
        generate_mock_data().to_csv(DATA_DIR / "ieee_ev_charging.csv", index=False)
    frame = add_multitarget_features(load_multitarget_data())
    range_bundle = _fit(frame, RANGE_FEATURES, "range_km", RANGE_MODEL_PATH)
    # SoH is the primary target; annual loss is retained as a transparent derived estimate.
    health_bundle = _fit(frame, HEALTH_FEATURES, "soh_percentage", DEGRADATION_MODEL_PATH)
    return range_bundle, health_bundle

if __name__ == "__main__": train_models()
