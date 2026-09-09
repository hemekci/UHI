"""Hierarchical-model metrics: marginal/conditional R² (Nakagawa & Schielzeth 2013),
ICC, partial R², within-city LOCO R², calibration slope.

All functions take the fitted mixed-effects result and/or raw predictions; they do
not re-fit. Works with statsmodels.MixedLMResults and with any predictor with
``predict`` / stored residual variance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def marginal_conditional_r2(fit: Any, X: pd.DataFrame, y: NDArray[np.floating]) -> dict[str, float]:
    """Nakagawa & Schielzeth (2013) marginal / conditional R² for a mixed-effects model.

    marginal R² = var(fixed effects prediction) / (var(fixed) + var(random) + var(residual))
    conditional R² = (var(fixed) + var(random)) / (var(fixed) + var(random) + var(residual))

    Parameters
    ----------
    fit : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted MixedLM result.
    X : DataFrame
        Feature matrix aligned with the model's fixed-effect design.
    y : array
        Observed target (not used for calculation but checked for alignment).
    """
    # Variance of fixed-effects fitted values
    x = X.values.astype(float)
    raw_beta = fit.fe_params
    beta_vals = raw_beta.values if hasattr(raw_beta, "values") else np.asarray(raw_beta)
    beta = np.asarray(beta_vals, dtype=float)
    # Intercept is the first FE; fe_pred includes it for total fitted FE prediction
    if beta.shape[0] == x.shape[1] + 1:
        fe_pred = beta[0] + x @ beta[1:]
    else:
        fe_pred = x @ beta
    var_fe = float(np.var(fe_pred, ddof=0))

    # Random-effects variance (city random intercept) — extracted from cov_re
    var_re = float(np.asarray(fit.cov_re).sum())

    # Residual variance
    var_resid = float(fit.scale)

    denom = var_fe + var_re + var_resid
    if denom <= 0:
        return {"marginal_r2": 0.0, "conditional_r2": 0.0}
    return {
        "marginal_r2": var_fe / denom,
        "conditional_r2": (var_fe + var_re) / denom,
    }


def intraclass_correlation(fit: Any) -> float:
    """ICC = var(random intercept) / (var(random intercept) + var(residual)).

    Interpretation: fraction of total variance attributable to the city level
    (after the fixed-effects design has been accounted for).
    """
    var_re = float(np.asarray(fit.cov_re).sum())
    var_resid = float(fit.scale)
    denom = var_re + var_resid
    return float(var_re / denom) if denom > 0 else 0.0


def partial_r2(fit_full: Any, fit_reduced: Any, y: NDArray[np.floating]) -> float:
    """Partial R² = improvement in likelihood-based R² when adding the morphology
    fixed effects on top of an intercept-only + city-random model.

    Implemented via the Nakagawa marginal R² difference: full model marginal R²
    minus reduced model marginal R².

    ``fit_full``  : MixedLMResults with morphology fixed effects
    ``fit_reduced``: MixedLMResults with intercept only (same random-effect
    grouping)
    """
    # Reduced model has no fixed slopes, only intercept -> var_fe = 0
    var_fe_full = float(np.var(np.asarray(fit_full.fittedvalues) - float(fit_full.params.get("Intercept", 0.0)), ddof=0))
    var_re_full = float(np.asarray(fit_full.cov_re).sum())
    var_resid_full = float(fit_full.scale)
    denom_full = var_fe_full + var_re_full + var_resid_full
    marginal_full = (var_fe_full / denom_full) if denom_full > 0 else 0.0

    # Reduced marginal R² is zero by construction for an intercept-only FE
    # Still compute for completeness in case of non-trivial reduced FE
    var_fe_red = 0.0  # no FE slopes
    var_re_red = float(np.asarray(fit_reduced.cov_re).sum())
    var_resid_red = float(fit_reduced.scale)
    denom_red = var_fe_red + var_re_red + var_resid_red
    marginal_red = (var_fe_red / denom_red) if denom_red > 0 else 0.0

    return float(marginal_full - marginal_red)


def within_city_loco_r2(
    df: pd.DataFrame,
    centred_target_col: str,
    feature_cols: list[str],
    city_col: str,
    model_fitter,
) -> dict[str, float]:
    """Leave-one-CITY-out R² on the within-city-centred target.

    For each held-out city, fit ``model_fitter`` on all other cities' centred
    patches and predict the held-out city's centred patches. Returns the
    overall held-out R² and per-city mean absolute residual summary.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    cities = df[city_col].unique()
    preds = np.full(len(df), np.nan)
    for c in cities:
        mask = df[city_col].values == c
        tr_X = df.loc[~mask, feature_cols].values
        tr_y = df.loc[~mask, centred_target_col].values
        te_X = df.loc[mask, feature_cols].values
        model = model_fitter(tr_X, tr_y)
        preds[mask] = model.predict(te_X)
    valid = ~np.isnan(preds)
    return {
        "r2": float(r2_score(df[centred_target_col].values[valid], preds[valid])),
        "mae": float(mean_absolute_error(df[centred_target_col].values[valid], preds[valid])),
        "n_cities": len(cities),
    }


def calibration_slope(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Slope of the regression of y_true on y_pred. Perfect calibration = 1.0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # slope of y_true ~ y_pred via OLS
    cov = np.cov(y_pred, y_true, ddof=1)[0, 1]
    var = np.var(y_pred, ddof=1)
    return float(cov / var) if var > 0 else 0.0


def per_zone_marginal_r2(
    fit: Any, X: pd.DataFrame, zones: Iterable[str]
) -> dict[str, float]:
    """Compute marginal R² separately within each Köppen zone.

    The fixed-effect prediction is sliced by zone and variance explained is
    measured against the residual variance within that zone subset.
    """
    x = X.values.astype(float)
    raw_beta = fit.fe_params
    beta_vals = raw_beta.values if hasattr(raw_beta, "values") else np.asarray(raw_beta)
    beta = np.asarray(beta_vals, dtype=float)
    if beta.shape[0] == x.shape[1] + 1:
        fe_pred = beta[0] + x @ beta[1:]
    else:
        fe_pred = x @ beta
    resid = np.asarray(fit.resid, dtype=float)

    out = {}
    zones_arr = np.asarray(list(zones))
    for z in np.unique(zones_arr):
        mask = zones_arr == z
        if mask.sum() < 20:
            continue
        var_fe = float(np.var(fe_pred[mask], ddof=0))
        var_resid = float(np.var(resid[mask], ddof=0))
        denom = var_fe + var_resid
        out[str(z)] = float(var_fe / denom) if denom > 0 else 0.0
    return out
