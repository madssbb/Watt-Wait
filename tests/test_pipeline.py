"""Fast unit tests for the core data and decision policy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_ev_sessions
from src.feature_engineering import add_session_features, build_live_features
from src.nudge_logic import nudge_priority_score, risk_from_pcdr
from src.strategy_engine import recommend_target_soc


def _raw_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s2", "bad"],
            "user_id": ["u1", "u1", "u2"],
            "station_id": ["a", "a", "b"],
            "connection_start": ["2025-01-06 08:00", "2025-01-07 09:00", "2025-01-07 11:00"],
            "connection_end": ["2025-01-06 12:00", "2025-01-07 14:00", "2025-01-07 10:00"],
            "charging_start": ["2025-01-06 08:00", "2025-01-07 09:00", "2025-01-07 11:00"],
            "charging_end": ["2025-01-06 10:00", "2025-01-07 12:00", "2025-01-07 12:00"],
            "energy_kwh": [14.0, 21.0, 5.0],
            "user_type": ["Employee"] * 3,
            "department": ["Engineering"] * 3,
        }
    )


def test_loader_filters_invalid_and_computes_pcdr(tmp_path) -> None:
    path = tmp_path / "sessions.csv"
    _raw_sessions().to_csv(path, index=False)
    loaded = load_ev_sessions(path)
    assert len(loaded) == 2
    assert loaded.iloc[0]["pcdr"] == 0.5
    assert loaded.iloc[1]["pcdr"] == 0.4


def test_user_history_is_shifted_without_current_target_leakage(tmp_path) -> None:
    path = tmp_path / "sessions.csv"
    _raw_sessions().to_csv(path, index=False)
    featured = add_session_features(load_ev_sessions(path), global_prior=0.25)
    assert featured.iloc[0]["user_avg_pcdr"] == 0.25
    assert featured.iloc[1]["user_avg_pcdr"] == 0.5


def test_risk_and_priority_boundaries() -> None:
    assert risk_from_pcdr(0.1499) == "Low"
    assert risk_from_pcdr(0.15) == "Medium"
    assert risk_from_pcdr(0.40) == "High"
    assert nudge_priority_score(0.5, 0.75) == pytest.approx(0.6)


def test_live_features_use_known_profiles() -> None:
    row = build_live_features(
        user_id="u1",
        station_id="s1",
        arrival_time=pd.Timestamp("2025-01-06 08:30"),
        energy_kwh=20.0,
        user_type="Employee",
        department="Engineering",
        user_profiles={"u1": 0.42},
        station_rates={"s1": 7.2},
        global_pcdr=0.3,
        global_rate_kw=6.6,
    )
    assert row.at[0, "user_avg_pcdr"] == 0.42
    assert row.at[0, "avg_charging_rate_kw"] == 7.2
    assert row.at[0, "is_workplace_peak_arrival"] == 1


def test_strategy_only_uses_full_charge_for_range_shortfall() -> None:
    assert recommend_target_soc(300, 200, "Low", 95, 0.1)[0] == 100
    assert recommend_target_soc(100, 200, "High", 95, 0.1)[0] == 80
