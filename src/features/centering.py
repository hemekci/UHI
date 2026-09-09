"""Within-city centring helpers for hierarchical UHI analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def within_city_center(
    df: pd.DataFrame, target: str = "uhi_anomaly_c", city: str = "city_id"
) -> pd.Series:
    """Subtract per-city median from ``target``.

    The within-city centred target isolates patch-level variation from
    city-level climate/context offsets. Using median (not mean) is robust
    to within-city outliers.
    """
    medians = df.groupby(city)[target].transform("median")
    return df[target] - medians


def within_city_zscore(
    df: pd.DataFrame, columns: list[str], city: str = "city_id"
) -> pd.DataFrame:
    """Z-score a set of columns within each city group.

    Used to make archetype clustering express relative-to-own-city patterns
    rather than absolute morphology scales.
    """
    out = df.copy()
    for c in columns:
        g = out.groupby(city)[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        out[c] = (out[c] - mu) / sd
        out[c] = out[c].fillna(0.0)
    return out
