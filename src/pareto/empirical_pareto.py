"""Non-dominated sort on observed patches per Köppen zone.

Objectives are taken from the Hydra `pareto` config; sign=+1 means minimize,
sign=−1 means maximize (we negate internally so all objectives are minimized).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParetoResult:
    """Per-zone empirical Pareto front."""

    zone: str
    indices: np.ndarray          # indices into original DataFrame
    objective_values: np.ndarray # (n_pareto, n_obj) minimization-framed
    n_total: int
    n_pareto: int


def _is_non_dominated(F: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows in F (minimization)."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(F[i] >= F, axis=1) & np.any(F[i] > F, axis=1)
        if dominates_i.any():
            keep[i] = False
    return keep


def extract_pareto(
    df: pd.DataFrame,
    objectives: Sequence[dict],
    group_by: str = "koppen_zone",
    min_points_per_zone: int = 30,
) -> list[ParetoResult]:
    """Extract per-zone empirical Pareto fronts.

    Parameters
    ----------
    df : DataFrame
        Must contain columns referenced by `objectives` and by `group_by`.
    objectives : sequence of dict
        Each dict has keys 'key' (column name) and 'sign' (+1 minimize, -1 maximize).
    """
    keys = [o["key"] for o in objectives]
    signs = np.array([o["sign"] for o in objectives], dtype=float)
    results: list[ParetoResult] = []

    for zone, sub in df.groupby(group_by):
        if len(sub) < min_points_per_zone:
            continue
        F = sub[keys].values * signs  # minimization-framed
        mask = _is_non_dominated(F)
        pareto_idx = sub.index.values[mask]
        results.append(
            ParetoResult(
                zone=str(zone),
                indices=pareto_idx,
                objective_values=F[mask],
                n_total=len(sub),
                n_pareto=int(mask.sum()),
            )
        )
    return results
