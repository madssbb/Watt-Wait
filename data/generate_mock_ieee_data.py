"""Generate reproducible, interconnected IEEE DataPort-style EV datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def generate_mock_data(n_sessions: int = 2_500, seed: int = 42) -> pd.DataFrame:
    """Return legacy-compatible workplace charging records with peak power."""
    if n_sessions < 100:
        raise ValueError("n_sessions must be at least 100")
    rng = np.random.default_rng(seed)
    users = np.array([f"USR-{i:04d}" for i in range(1, 181)])
    stations = np.array([f"STN-{i:02d}" for i in range(1, 17)])
    habits = dict(zip(users, rng.beta(2.2, 3.2, len(users))))
    start = pd.Timestamp("2024-01-02")
    rows = []
    for i in range(n_sessions):
        user, station = str(rng.choice(users)), str(rng.choice(stations))
        day = start + pd.Timedelta(days=int(rng.integers(0, 540)))
        hour = float(np.clip(rng.normal(8.6 if day.dayofweek < 5 else 11, 1.7), 5.5, 20))
        connected = day + pd.Timedelta(hours=hour)
        energy = float(np.clip(rng.gamma(3, 6), 3, 65))
        peak = float(rng.choice([6.6, 7.2, 11.0, 19.2], p=[.36, .39, .17, .08]))
        active = max(.35, energy / (peak * rng.uniform(.76, .97)) + rng.normal(.08, .08))
        charging_start = connected + pd.Timedelta(minutes=int(rng.integers(0, 20)))
        charging_end = charging_start + pd.Timedelta(hours=active)
        pcdr = float(np.clip(habits[user] * .72 + (.09 if 7 <= hour < 10 else -.04) + rng.normal(0, .10), .01, .88))
        end = charging_end + pd.Timedelta(hours=active * pcdr / max(1 - pcdr, .05))
        rows.append(dict(session_id=f"SES-{i+1:06d}", user_id=user, station_id=station,
            connection_start=connected.round("s"), connection_end=end.round("s"),
            charging_start=charging_start.round("s"), charging_end=charging_end.round("s"),
            energy_kwh=round(energy, 3), peak_power_kw=peak,
            user_type="Employee", department="Operations"))
    return pd.DataFrame(rows).sort_values("connection_start").reset_index(drop=True)


def generate_all(n_sessions: int = 2_500, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create charging, trip telemetry, and battery-health tables sharing users/vehicles."""
    charging = generate_mock_data(n_sessions, seed)
    rng = np.random.default_rng(seed + 1)
    users = sorted(charging.user_id.unique())
    vehicle_map = {user: f"VEH-{i:04d}" for i, user in enumerate(users, 1)}
    vehicles = pd.DataFrame({"user_id": users, "vehicle_id": [vehicle_map[u] for u in users]})
    telemetry = charging[["session_id", "user_id"]].copy()
    telemetry["vehicle_id"] = telemetry.user_id.map(vehicle_map)
    telemetry["trip_distance_km"] = np.round(rng.uniform(12, 220, len(telemetry)), 2)
    telemetry["avg_speed_kmph"] = np.round(rng.uniform(25, 105, len(telemetry)), 2)
    # Normal-car trip dynamics: m/s² acceleration, event count, idle %, and
    # normalized throttle variance. These are synthetic IEEE-style telemetry.
    telemetry["max_acceleration_ms2"] = np.round(rng.uniform(.6, 3.4, len(telemetry)), 2)
    telemetry["hard_braking_count"] = rng.poisson(1.2, len(telemetry))
    telemetry["idle_time_percentage"] = np.round(rng.uniform(0, 22, len(telemetry)), 2)
    telemetry["rpm_or_throttle_variance"] = np.round(rng.uniform(.05, .8, len(telemetry)), 3)
    telemetry["ambient_temp_c"] = np.round(rng.normal(25, 8, len(telemetry)).clip(-5, 45), 2)
    telemetry["hv_heating_ac_power_kw"] = np.round(np.maximum(.2, 1.3 + (20 - telemetry.ambient_temp_c).clip(lower=0)*.10 + rng.normal(0,.2,len(telemetry))), 3)
    efficiency = .145 + (telemetry.avg_speed_kmph - 60).abs()*.001 + (20-telemetry.ambient_temp_c).abs()*.002
    telemetry["energy_consumed_kwh"] = np.round(telemetry.trip_distance_km * efficiency, 3)
    cycles = rng.integers(30, 1800, len(vehicles)); fast = rng.uniform(.02, .92, len(vehicles)); temp = rng.normal(30, 5, len(vehicles)).clip(18, 52)
    degradation = (.7 + cycles*.0012 + fast*1.2 + np.maximum(temp-30, 0)*.06 + rng.normal(0,.12,len(vehicles))).clip(.3, 6)
    health = vehicles.copy(); health["battery_type"] = rng.choice(["NMC", "LFP"], len(health), p=[.7,.3])
    health["charge_cycles"], health["fast_charge_ratio"] = cycles, np.round(fast, 3); health["avg_battery_temp_c"] = np.round(temp, 2)
    health["degradation_rate_per_year"] = np.round(degradation, 3)
    health["soh_percentage"] = np.round((100 - cycles*.012 - fast*3 - np.maximum(temp-30,0)*.16 + rng.normal(0,1,len(health))).clip(65,100), 2)
    return {"workplace": charging.drop(columns=["charging_start", "user_type", "department"]), "telemetry": telemetry, "health": health}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--rows", type=int, default=2500); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); tables = generate_all(args.rows, args.seed)
    paths = {"workplace": DATA_DIR/"ieee_workplace_charging.csv", "telemetry": DATA_DIR/"ieee_vehicle_telemetry.csv", "health": DATA_DIR/"ieee_battery_health.csv"}
    for name, table in tables.items(): table.to_csv(paths[name], index=False)
    generate_mock_data(args.rows, args.seed).to_csv(DATA_DIR/"ieee_ev_charging.csv", index=False)
    print("Generated " + ", ".join(str(p) for p in paths.values()))

if __name__ == "__main__": main()
