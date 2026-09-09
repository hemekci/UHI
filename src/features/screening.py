"""Physical-plausibility quality screen for the surface-UHI target.

A summer surface-UHI anomaly of several tens of degrees below the rural
reference is not a thermal signal but a residual cloud or fill artifact,
concentrated in the cloudiest Koppen zones. This module removes such patches
before any modelling, Pareto extraction, or clustering, so that downstream
coefficients are not driven by cloud contamination.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_UHI_ABS_MAX_C: float = 40.0


def quality_screen(
    df: pd.DataFrame,
    abs_max_c: float = DEFAULT_UHI_ABS_MAX_C,
    target_col: str = "uhi_anomaly_c",
) -> pd.DataFrame:
    """Drop patches whose absolute UHI anomaly exceeds ``abs_max_c``.

    Args:
        df: Patch-level feature/target table.
        abs_max_c: Maximum physically plausible absolute UHI anomaly (Celsius).
        target_col: Name of the UHI-anomaly column.

    Returns:
        A filtered copy of ``df`` containing only physically plausible patches.
    """
    if target_col not in df.columns:
        logger.warning("Quality screen skipped: column %s not found", target_col)
        return df
    n_before = len(df)
    mask = df[target_col].abs() <= abs_max_c
    out = df.loc[mask].copy()
    n_dropped = n_before - len(out)
    logger.info(
        "Quality screen |%s| <= %.1f C: dropped %d/%d patches (%.2f%%)",
        target_col, abs_max_c, n_dropped, n_before,
        100.0 * n_dropped / n_before if n_before else 0.0,
    )
    return out
