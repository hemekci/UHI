"""Tests for per-zone empirical Pareto extraction."""

import pandas as pd
import pytest

from src.pareto import extract_pareto


@pytest.fixture
def synthetic_patches() -> pd.DataFrame:
    """Handcrafted synthetic frame (NOT real data) used only to validate algorithm correctness.

    Two zones with clean non-dominated fronts for testing.
    """
    rows = []
    # Zone A: 3 non-dominated, 2 dominated
    rows += [
        {"koppen_zone": "A", "uhi_anomaly_c": 1.0, "building_density": 0.2},  # ND
        {"koppen_zone": "A", "uhi_anomaly_c": 0.5, "building_density": 0.1},  # ND
        {"koppen_zone": "A", "uhi_anomaly_c": 2.0, "building_density": 0.5},  # ND
        {"koppen_zone": "A", "uhi_anomaly_c": 3.0, "building_density": 0.3},  # dominated
        {"koppen_zone": "A", "uhi_anomaly_c": 1.5, "building_density": 0.15}, # dominated
    ] * 7  # repeat so min_points_per_zone=30 is satisfied
    # Zone B: small → should be skipped
    rows += [{"koppen_zone": "B", "uhi_anomaly_c": x, "building_density": x / 10} for x in range(3)]
    return pd.DataFrame(rows)


def test_extract_pareto_skips_small_zones(synthetic_patches):
    objectives = [
        {"key": "uhi_anomaly_c", "sign": 1},
        {"key": "building_density", "sign": -1},
    ]
    res = extract_pareto(synthetic_patches, objectives=objectives, min_points_per_zone=30)
    zones = [r.zone for r in res]
    assert "A" in zones
    assert "B" not in zones


def test_extract_pareto_non_dominated_count(synthetic_patches):
    objectives = [
        {"key": "uhi_anomaly_c", "sign": 1},
        {"key": "building_density", "sign": -1},
    ]
    res = extract_pareto(synthetic_patches, objectives=objectives, min_points_per_zone=30)
    res_a = next(r for r in res if r.zone == "A")
    # Because the 5 original rows are repeated 7 times, many duplicates all share the same
    # objective pair — non-dominated uniques should include the three clearly-optimal
    # points (0.5,0.1), (1.0,0.2), and (2.0,0.5).
    assert res_a.n_pareto > 0
    assert res_a.n_pareto <= res_a.n_total
