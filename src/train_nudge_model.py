"""Train Stage 2 risk and nudge-responsiveness classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import DEFAULT_DATA_PATH, load_ev_sessions
from src.feature_engineering import MODEL_FEATURES, add_session_features
from src.nudge_logic import NUDGE_FEATURES, estimate_training_congestion, risk_series
from src.train_pcdr_model import DEFAULT_MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUDGE_MODEL_PATH = PROJECT_ROOT / "models" / "nudge_classifier.pkl"


def train_nudge_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    pcdr_model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = DEFAULT_NUDGE_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit risk and response models and persist their policy in one bundle.

    Because the source schema has no notification outcome, responsiveness uses
    a documented behavioral proxy: departure PCDR at least 20% better than the
    driver's prior mean (or below 0.15). Replace this label with observed nudge
    outcomes before production use.
    """
    print(f"[1/5] Loading PCDR model from {pcdr_model_path}")
    pcdr_bundle = joblib.load(pcdr_model_path)
    sessions = load_ev_sessions(data_path)
    prior = float(pcdr_bundle["global_pcdr"])
    featured = add_session_features(sessions, global_prior=prior)
    featured["predicted_pcdr"] = np.clip(
        pcdr_bundle["pipeline"].predict(featured[MODEL_FEATURES]), 0.0, 1.0
    )
    featured["station_congestion_factor"] = estimate_training_congestion(featured)
    featured["risk_level"] = risk_series(featured["pcdr"])
    featured["response_proxy"] = (
        featured["pcdr"] <= np.maximum(0.15, featured["user_avg_pcdr"] * 0.80)
    ).astype(int)

    split_index = int(len(featured) * 0.80)
    train = featured.iloc[:split_index]
    test = featured.iloc[split_index:]
    print(f"[2/5] Prepared {len(train):,} train / {len(test):,} test rows")

    risk_model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    print("[3/5] Fitting risk classifier")
    risk_model.fit(train[NUDGE_FEATURES], train["risk_level"])

    responsiveness_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1_000, random_state=random_state
                ),
            ),
        ]
    )
    print("[4/5] Fitting responsiveness-proxy classifier")
    responsiveness_model.fit(train[NUDGE_FEATURES], train["response_proxy"])

    risk_prediction = risk_model.predict(test[NUDGE_FEATURES])
    response_probability = responsiveness_model.predict_proba(test[NUDGE_FEATURES])[:, 1]
    metrics: dict[str, Any] = {
        "risk_accuracy": float(accuracy_score(test["risk_level"], risk_prediction)),
        "risk_classification_report": classification_report(
            test["risk_level"], risk_prediction, output_dict=True, zero_division=0
        ),
        "responsiveness_proxy_positive_rate": float(train["response_proxy"].mean()),
    }
    if test["response_proxy"].nunique() == 2:
        metrics["responsiveness_proxy_roc_auc"] = float(
            roc_auc_score(test["response_proxy"], response_probability)
        )

    bundle = {
        "risk_classifier": risk_model,
        "responsiveness_classifier": responsiveness_model,
        "features": NUDGE_FEATURES,
        "metrics": metrics,
        "risk_thresholds": {"low_max_exclusive": 0.15, "medium_max_exclusive": 0.40},
        "priority_weights": {"predicted_pcdr": 0.60, "station_congestion": 0.40},
        "alert_policy": {"risk": "High", "congestion_strictly_greater_than": 0.70},
        "responsiveness_label": "behavioral_proxy_not_observed_nudge_response",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"[5/5] Saved nudge bundle to {output_path}")
    print(json.dumps({key: value for key, value in metrics.items() if key != "risk_classification_report"}, indent=2))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--pcdr-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_NUDGE_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_nudge_model(args.data, args.pcdr_model, args.output, args.seed)


if __name__ == "__main__":
    main()
