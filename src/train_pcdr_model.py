"""Train and persist the Stage 1 PCDR random-forest regression pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import DEFAULT_DATA_PATH, load_ev_sessions
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    add_session_features,
    station_rate_profiles,
    user_pcdr_profiles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pcdr_regressor.pkl"
DEFAULT_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "pcdr_feature_importance.csv"


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the preprocessing and random-forest regression pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessing), ("regressor", regressor)])


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    values = pipeline.named_steps["regressor"].feature_importances_
    detailed = pd.DataFrame({"feature": names, "importance": values})
    # Combine one-hot levels so the dashboard remains readable.
    def parent_feature(name: str) -> str:
        for categorical in CATEGORICAL_FEATURES:
            if name.startswith(f"{categorical}_"):
                return categorical
        return name

    detailed["feature"] = detailed["feature"].map(parent_feature)
    return (
        detailed.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_pcdr_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    importance_path: str | Path = DEFAULT_IMPORTANCE_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train with a chronological 80/20 holdout and save a deployment bundle."""
    print(f"[1/5] Loading sessions from {data_path}")
    sessions = load_ev_sessions(data_path)
    if len(sessions) < 100:
        raise ValueError("At least 100 valid sessions are required for training")

    split_index = int(len(sessions) * 0.80)
    training_prior = float(sessions.iloc[:split_index]["pcdr"].mean())
    featured = add_session_features(sessions, global_prior=training_prior)
    train = featured.iloc[:split_index]
    test = featured.iloc[split_index:]
    print(f"[2/5] Chronological split: {len(train):,} train / {len(test):,} test")

    pipeline = build_pipeline(random_state)
    print("[3/5] Fitting RandomForestRegressor")
    pipeline.fit(train[MODEL_FEATURES], train["pcdr"])
    predictions = np.clip(pipeline.predict(test[MODEL_FEATURES]), 0.0, 1.0)
    metrics = {
        "rmse": float(mean_squared_error(test["pcdr"], predictions) ** 0.5),
        "mae": float(mean_absolute_error(test["pcdr"], predictions)),
        "r2": float(r2_score(test["pcdr"], predictions)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    print(f"[4/5] Metrics: {json.dumps(metrics, indent=2)}")

    importance = _feature_importance(pipeline)
    model_path = Path(model_path)
    importance_path = Path(importance_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)

    rates = station_rate_profiles(sessions)
    global_rate = float(np.median(list(rates.values())))
    bundle = {
        "pipeline": pipeline,
        "model_features": MODEL_FEATURES,
        "metrics": metrics,
        "trained_through": str(sessions["connection_start"].max()),
        "global_pcdr": float(sessions["pcdr"].mean()),
        "global_rate_kw": global_rate,
        "user_profiles": user_pcdr_profiles(sessions),
        "station_rates": rates,
        "user_types": sorted(sessions["user_type"].dropna().astype(str).unique().tolist()),
        "departments": sorted(sessions["department"].dropna().astype(str).unique().tolist()),
        "feature_importance": importance.to_dict(orient="records"),
        "model_name": "RandomForestRegressor",
        "target": "pcdr",
    }
    joblib.dump(bundle, model_path)
    importance.to_csv(importance_path, index=False)
    print(f"[5/5] Saved model to {model_path}")
    print("Top feature importances:")
    print(importance.head(10).to_string(index=False))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--importance", type=Path, default=DEFAULT_IMPORTANCE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_pcdr_model(args.data, args.model, args.importance, args.seed)


if __name__ == "__main__":
    main()

