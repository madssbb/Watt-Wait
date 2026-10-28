"""Leakage-safe feature engineering for PCDR and nudge models."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


NUMERIC_FEATURES: Final = [
    "arrival_hour",
    "arrival_dayofweek",
    "is_weekend",
    "is_workplace_peak_arrival",
    "energy_kwh",
    "avg_charging_rate_kw",
    "user_avg_pcdr",
]
CATEGORICAL_FEATURES: Final = ["station_id", "user_type", "department"]
MODEL_FEATURES: Final = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_session_features(frame: pd.DataFrame, global_prior: float | None = None) -> pd.DataFrame:
    """Add model features, using only earlier sessions for each user's PCDR mean.

    The frame is chronologically sorted before the shifted expanding average is
    calculated. First-time users receive the supplied global prior. During model
    training, callers should omit it so only the training frame's target mean is
    used as the fallback.
    """
    required = {
        "session_id",
        "connection_start",
        "energy_kwh",
        "active_duration_hrs",
        "user_id",
        "station_id",
        "user_type",
        "department",
        "pcdr",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot engineer features; missing columns: {sorted(missing)}")

    result = frame.sort_values(["connection_start", "session_id"]).copy()
    result["arrival_hour"] = result["connection_start"].dt.hour + (
        result["connection_start"].dt.minute / 60.0
    )
    result["arrival_dayofweek"] = result["connection_start"].dt.dayofweek
    result["is_weekend"] = (result["arrival_dayofweek"] >= 5).astype(int)
    result["is_workplace_peak_arrival"] = (
        (result["arrival_hour"] >= 7.0) & (result["arrival_hour"] < 10.0)
    ).astype(int)
    result["avg_charging_rate_kw"] = np.divide(
        result["energy_kwh"],
        result["active_duration_hrs"],
        out=np.zeros(len(result), dtype=float),
        where=result["active_duration_hrs"].to_numpy() > 0,
    )

    historical = result.groupby("user_id", sort=False)["pcdr"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    prior = float(result["pcdr"].mean()) if global_prior is None else float(global_prior)
    result["user_avg_pcdr"] = historical.fillna(prior).clip(0.0, 1.0)
    return result.reset_index(drop=True)


def user_pcdr_profiles(frame: pd.DataFrame) -> dict[str, float]:
    """Return latest historical mean PCDR per user for live inference."""
    return frame.groupby("user_id")["pcdr"].mean().clip(0.0, 1.0).astype(float).to_dict()


def station_rate_profiles(frame: pd.DataFrame) -> dict[str, float]:
    """Return robust median observed charging rate per station."""
    rates = frame.assign(
        avg_charging_rate_kw=frame["energy_kwh"] / frame["active_duration_hrs"]
    )
    rates = rates.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_charging_rate_kw"])
    return rates.groupby("station_id")["avg_charging_rate_kw"].median().astype(float).to_dict()


def build_live_features(
    *,
    user_id: str,
    station_id: str,
    arrival_time: pd.Timestamp,
    energy_kwh: float,
    user_type: str,
    department: str,
    user_profiles: dict[str, float],
    station_rates: dict[str, float],
    global_pcdr: float,
    global_rate_kw: float,
) -> pd.DataFrame:
    """Build a one-row feature frame for a newly connected session."""
    arrival = pd.Timestamp(arrival_time)
    hour = arrival.hour + arrival.minute / 60.0
    values = {
        "arrival_hour": hour,
        "arrival_dayofweek": arrival.dayofweek,
        "is_weekend": int(arrival.dayofweek >= 5),
        "is_workplace_peak_arrival": int(7.0 <= hour < 10.0),
        "energy_kwh": float(energy_kwh),
        "avg_charging_rate_kw": float(station_rates.get(station_id, global_rate_kw)),
        "user_avg_pcdr": float(user_profiles.get(user_id, global_pcdr)),
        "station_id": str(station_id),
        "user_type": str(user_type),
        "department": str(department),
    }
    return pd.DataFrame([values], columns=MODEL_FEATURES)


RANGE_FEATURES: Final = ["current_soc_pct", "battery_capacity_kwh", "efficiency_kwh_per_km", "ambient_temp_c", "avg_speed_kmph"]
HEALTH_FEATURES: Final = ["charge_cycles", "fast_charge_ratio", "avg_battery_temp_c", "avg_soc_pct", "thermal_stress_index"]

DRIVING_TELEMETRY_FEATURES: Final = [
    "avg_speed_kmph", "max_acceleration_ms2", "hard_braking_count",
    "idle_time_percentage", "rpm_or_throttle_variance",
]


def classify_driving_mode(telemetry: pd.Series | dict) -> str:
    """Classify a trip as Eco, Normal, or Sport from IEEE-style telemetry."""
    values = pd.Series(telemetry)
    missing = set(DRIVING_TELEMETRY_FEATURES) - set(values.index)
    if missing:
        raise ValueError(f"Driving telemetry missing columns: {sorted(missing)}")
    acceleration = float(values["max_acceleration_ms2"])
    braking = float(values["hard_braking_count"])
    variance = float(values["rpm_or_throttle_variance"])
    speed = float(values["avg_speed_kmph"])
    if acceleration > 2.5 or braking >= 3 or (speed >= 90 and variance >= 0.35):
        return "Sport"
    if acceleration < 1.5 and braking <= 1 and variance < 0.35:
        return "Eco"
    return "Normal"


def add_multitarget_features(frame: pd.DataFrame, battery_capacity_kwh: float = 75.0) -> pd.DataFrame:
    """Add explainable thermal, utilization, efficiency, and range target features."""
    result = frame.copy()
    result["thermal_stress_index"] = result["avg_battery_temp_c"] * result["fast_charge_ratio"]
    result["daily_utilization_intensity"] = result["trip_distance_km"] / result["charge_cycles"].clip(lower=1)
    result["efficiency_kwh_per_km"] = result["energy_consumed_kwh"] / result["trip_distance_km"].clip(lower=.1)
    # Synthetic current SOC represents a present-state reading, not a label leak.
    result["current_soc_pct"] = np.clip(72 - result["energy_consumed_kwh"] / battery_capacity_kwh * 100 + 25, 15, 100)
    result["battery_capacity_kwh"] = battery_capacity_kwh * result["soh_percentage"] / 100
    temp_factor = (1 - (result["ambient_temp_c"] - 22).abs() * .004).clip(.75, 1.0)
    result["range_km"] = result["current_soc_pct"] / 100 * result["battery_capacity_kwh"] / result["efficiency_kwh_per_km"].clip(lower=.08) * temp_factor
    result["endurance_hours"] = result["range_km"] / result["avg_speed_kmph"].clip(lower=10)
    # SOC maintenance is a controllable health signal in a production telemetry feed.
    result["avg_soc_pct"] = np.clip(65 + result["fast_charge_ratio"] * 15, 40, 90)
    return result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
