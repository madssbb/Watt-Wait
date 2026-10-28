"""Load, normalize, validate, and label IEEE-style EV charging sessions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "ieee_ev_charging.csv"
DEFAULT_WORKPLACE_PATH: Final = PROJECT_ROOT / "data" / "ieee_workplace_charging.csv"
DEFAULT_TELEMETRY_PATH: Final = PROJECT_ROOT / "data" / "ieee_vehicle_telemetry.csv"
DEFAULT_HEALTH_PATH: Final = PROJECT_ROOT / "data" / "ieee_battery_health.csv"

REQUIRED_COLUMNS: Final = {
    "session_id",
    "user_id",
    "station_id",
    "connection_start",
    "connection_end",
    "charging_start",
    "charging_end",
    "energy_kwh",
}
TIMESTAMP_COLUMNS: Final = [
    "connection_start",
    "connection_end",
    "charging_start",
    "charging_end",
]
ALIASES: Final = {
    "session": "session_id",
    "sessionid": "session_id",
    "user": "user_id",
    "userid": "user_id",
    "station": "station_id",
    "stationid": "station_id",
    "connect_start": "connection_start",
    "connect_time": "connection_start",
    "connection_time": "connection_start",
    "connect_end": "connection_end",
    "disconnect_time": "connection_end",
    "disconnect_timestamp": "connection_end",
    "charge_start": "charging_start",
    "charge_end": "charging_end",
    "kwh": "energy_kwh",
    "energy": "energy_kwh",
    "energy_delivered_kwh": "energy_kwh",
}


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common IEEE export column spellings to the canonical schema."""
    normalized = {_column: _snake_case(str(_column)) for _column in frame.columns}
    result = frame.rename(columns=normalized)
    result = result.rename(columns={column: ALIASES.get(column, column) for column in result.columns})
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Columns map to duplicate canonical names: {duplicates}")
    return result


def load_ev_sessions(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV and return valid sessions with duration and PCDR labels.

    Invalid rows are removed when timestamps are missing or inconsistent,
    durations are non-positive, charging falls outside the connection window,
    or energy is negative/non-numeric.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python data/generate_mock_ieee_data.py"
        )

    frame = normalize_columns(pd.read_csv(path, low_memory=False))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["energy_kwh"] = pd.to_numeric(frame["energy_kwh"], errors="coerce")

    for column in ("session_id", "user_id", "station_id"):
        frame[column] = frame[column].astype("string").str.strip()
    for optional in ("user_type", "department"):
        if optional not in frame:
            frame[optional] = "Unknown"
        frame[optional] = frame[optional].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    frame["total_duration_hrs"] = (
        frame["connection_end"] - frame["connection_start"]
    ).dt.total_seconds() / 3_600
    frame["active_duration_hrs"] = (
        frame["charging_end"] - frame["charging_start"]
    ).dt.total_seconds() / 3_600

    valid = (
        frame[TIMESTAMP_COLUMNS].notna().all(axis=1)
        & frame[["session_id", "user_id", "station_id"]].notna().all(axis=1)
        & frame["energy_kwh"].notna()
        & (frame["energy_kwh"] >= 0)
        & (frame["total_duration_hrs"] > 0)
        & (frame["active_duration_hrs"] > 0)
        & (frame["charging_start"] >= frame["connection_start"])
        & (frame["charging_end"] <= frame["connection_end"])
        & (frame["active_duration_hrs"] <= frame["total_duration_hrs"])
    )
    frame = frame.loc[valid].copy()
    frame["pcdr"] = np.clip(
        (frame["total_duration_hrs"] - frame["active_duration_hrs"])
        / frame["total_duration_hrs"],
        0.0,
        1.0,
    )

    if frame.empty:
        raise ValueError("No valid charging sessions remain after validation")

    return frame.sort_values(["connection_start", "session_id"]).reset_index(drop=True)


def load_multitarget_data(
    workplace_path: str | Path = DEFAULT_WORKPLACE_PATH,
    telemetry_path: str | Path = DEFAULT_TELEMETRY_PATH,
    health_path: str | Path = DEFAULT_HEALTH_PATH,
) -> pd.DataFrame:
    """Load and join IEEE-style charging, driving, and battery health datasets.

    The telemetry generator provides ``session_id`` for an unambiguous session join;
    external files without it are joined by user/vehicle using their latest record.
    """
    workplace = normalize_columns(pd.read_csv(workplace_path))
    telemetry = normalize_columns(pd.read_csv(telemetry_path))
    health = normalize_columns(pd.read_csv(health_path))
    for name, frame, fields in [("workplace", workplace, {"session_id", "user_id", "connection_start", "connection_end", "charging_end", "energy_kwh"}), ("telemetry", telemetry, {"vehicle_id", "user_id", "trip_distance_km", "energy_consumed_kwh"}), ("health", health, {"vehicle_id", "charge_cycles", "fast_charge_ratio", "avg_battery_temp_c", "soh_percentage", "degradation_rate_per_year"})]:
        missing = fields - set(frame.columns)
        if missing: raise ValueError(f"{name} dataset missing columns: {sorted(missing)}")
    for col in ("connection_start", "connection_end", "charging_end"):
        workplace[col] = pd.to_datetime(workplace[col], errors="coerce")
    workplace["connection_duration_hrs"] = (workplace.connection_end - workplace.connection_start).dt.total_seconds() / 3600
    workplace["charging_duration_hrs"] = (workplace.charging_end - workplace.connection_start).dt.total_seconds() / 3600
    workplace["pcdr"] = ((workplace.connection_duration_hrs - workplace.charging_duration_hrs) / workplace.connection_duration_hrs).clip(0, 1)
    numeric = ["trip_distance_km", "avg_speed_kmph", "ambient_temp_c", "hv_heating_ac_power_kw", "energy_consumed_kwh"]
    telemetry[numeric] = telemetry[numeric].apply(pd.to_numeric, errors="coerce")
    health_cols = ["charge_cycles", "fast_charge_ratio", "avg_battery_temp_c", "soh_percentage", "degradation_rate_per_year"]
    health[health_cols] = health[health_cols].apply(pd.to_numeric, errors="coerce")
    merged = workplace.merge(telemetry, on=["session_id", "user_id"], how="inner").merge(health, on="vehicle_id", how="inner", suffixes=("", "_health"))
    return merged.dropna().sort_values("connection_start").reset_index(drop=True)


if __name__ == "__main__":
    sessions = load_ev_sessions()
    print(f"Loaded {len(sessions):,} valid sessions")
    print(sessions[["active_duration_hrs", "total_duration_hrs", "pcdr"]].describe())
