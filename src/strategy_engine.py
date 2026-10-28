"""Auditable charging-target policy combining range, health, and congestion."""
from __future__ import annotations

def recommend_target_soc(daily_range_km: float, predicted_range_at_80: float, dwell_risk: str, soh_pct: float, fast_charge_ratio: float) -> tuple[int, str]:
    """Return 80/100 SOC cutoff and a human-readable rationale.

    100% is reserved for a genuine range shortfall; 80% protects battery life and
    frees workplace stations when range is sufficient.
    """
    needs_full = daily_range_km > predicted_range_at_80 * .95
    health_sensitive = soh_pct < 82 or fast_charge_ratio >= .5
    if needs_full:
        return 100, "Daily range need exceeds the 80% buffer; charge to 100% for trip feasibility."
    if health_sensitive or dwell_risk == "High":
        return 80, "80% meets range needs and reduces battery stress or station dwell pressure."
    return 80, "80% comfortably meets expected travel while preserving battery health."
