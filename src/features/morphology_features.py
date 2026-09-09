"""Deterministic morphology-feature builders over per-patch building collections.

These functions are pure: they take building geometries and heights, and return
scalar features. They are independent of any GIS framework beyond geopandas/shapely.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def building_density(
    total_footprint_m2: float, patch_area_m2: float
) -> float:
    """Ratio of summed footprint area to patch area (0–1)."""
    if patch_area_m2 <= 0:
        return 0.0
    return max(0.0, min(1.0, total_footprint_m2 / patch_area_m2))


def height_mean_cv(
    heights_m: Iterable[float], weights: Iterable[float] | None = None
) -> tuple[float, float]:
    """Return (area-weighted mean height, coefficient of variation).

    If weights is None, equal weighting is used.
    CV = std / mean. Returns (0.0, 0.0) for empty inputs.
    """
    h = np.asarray(list(heights_m), dtype=float)
    if h.size == 0:
        return 0.0, 0.0
    w = np.asarray(list(weights), dtype=float) if weights is not None else np.ones_like(h)
    if w.sum() <= 0:
        return 0.0, 0.0
    w = w / w.sum()
    mean_h = float((h * w).sum())
    var_h = float((w * (h - mean_h) ** 2).sum())
    std_h = math.sqrt(max(0.0, var_h))
    cv = std_h / mean_h if mean_h > 0 else 0.0
    return mean_h, cv


def canyon_aspect_proxy(
    mean_height_m: float, median_inter_building_spacing_m: float
) -> float:
    """H/W proxy = mean building height / median pairwise inter-building spacing.

    Capped at 10 to tame outliers in very dense cores.
    """
    if median_inter_building_spacing_m <= 0:
        return 0.0
    return min(10.0, mean_height_m / median_inter_building_spacing_m)


def orientation_entropy(
    street_bearings_deg: Iterable[float], n_bins: int = 36
) -> float:
    """Shannon entropy (nats) of street-segment bearing distribution over [0, 180)."""
    bearings = np.asarray(list(street_bearings_deg), dtype=float)
    if bearings.size == 0:
        return 0.0
    bearings = np.mod(bearings, 180.0)
    hist, _ = np.histogram(bearings, bins=n_bins, range=(0.0, 180.0))
    p = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist.astype(float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
