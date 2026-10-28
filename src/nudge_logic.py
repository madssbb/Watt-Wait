"""Shared nudge scoring and risk policy used in training and the dashboard."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


RISK_ORDER: Final = ["Low", "Medium", "High"]
NUDGE_FEATURES: Final = [
    "predicted_pcdr",
    "station_congestion_factor",
    "user_avg_pcdr",
    "arrival_hour",
    "arrival_dayofweek",
    "is_weekend",
    "is_workplace_peak_arrival",
    "energy_kwh",
    "avg_charging_rate_kw",
]


def risk_from_pcdr(pcdr: float) -> str:
    """Map a PCDR prediction to the product's defined risk category."""
    if pcdr < 0.15:
        return "Low"
    if pcdr < 0.40:
        return "Medium"
    return "High"


def risk_series(pcdr: pd.Series) -> pd.Series:
    """Vectorized version of :func:`risk_from_pcdr`."""
    return pd.cut(
        pcdr.clip(0.0, 1.0),
        bins=[-np.inf, 0.15, 0.40, np.inf],
        labels=RISK_ORDER,
        right=False,
    ).astype(str)


def nudge_priority_score(predicted_pcdr: float, congestion_factor: float) -> float:
    """Calculate the required weighted PCDR/congestion priority score."""
    return float(np.clip(predicted_pcdr * 0.60 + congestion_factor * 0.40, 0.0, 1.0))


def smart_nudge_score(priority_score: float, responsiveness_probability: float) -> float:
    """Temper priority by predicted driver responsiveness without hiding urgency."""
    multiplier = 0.75 + 0.25 * float(np.clip(responsiveness_probability, 0.0, 1.0))
    return float(np.clip(priority_score * multiplier, 0.0, 1.0))


def estimate_training_congestion(frame: pd.DataFrame) -> pd.Series:
    """Estimate congestion from station/hour traffic when occupancy is unavailable.

    Production inference uses the live occupancy supplied by the dashboard. This
    proxy exists only because the source schema does not contain live occupancy.
    """
    keys = [frame["station_id"], frame["connection_start"].dt.floor("h")]
    arrivals_per_hour = frame.groupby(keys)["session_id"].transform("count").astype(float)
    scale = max(float(arrivals_per_hour.quantile(0.95)), 1.0)
    density = (arrivals_per_hour / scale).clip(0.0, 1.0)
    peak = frame["is_workplace_peak_arrival"].astype(float)
    return (0.65 * density + 0.35 * peak).clip(0.0, 1.0)

