"""Comprehensive hierarchical-model diagnostics — the paper's quantitative backbone.

Writes a single ``diagnostics.json`` with nine metric families on the configured
features parquet:

1. Variance decomposition: ICC, between-city / within-city variance shares
2. Mixed-effects marginal R², conditional R², partial R²
3. Per-zone marginal R²
4. Fixed-effect coefficients (beta, SE, 95% CI, p-value) for every morphology feature
5. Within-city LOCO R² on the centred target
6. Bootstrap SHAP top-5 stability (Jaccard) from within_city_xgboost
7. Zone-mean and zone-plus-city-mean baselines
8. Calibration slope
9. Per-patch prediction-interval coverage (80%)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import quality_screen, within_city_center  # noqa: E402
from src.metrics import (  # noqa: E402
    calibration_slope,
    intraclass_correlation,
    marginal_conditional_r2,
    within_city_loco_r2,
)
from src.metrics.hierarchical import per_zone_marginal_r2  # noqa: E402
from src.utils import get_logger, read_patches, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


# 8 morphology features (city-constant context features intentionally excluded).
# `impervious_frac = wc_built_frac + wc_bare_frac` and
# `vegetation_frac = wc_tree_frac + wc_shrub_frac + wc_grass_frac + wc_cropland_frac`
# are perfect linear combinations of the component WorldCover fractions; they
# are dropped to avoid collinearity-induced NaN confidence intervals.
MORPH_FEATS = [
    "building_density", "height_mean", "height_cv",
    "ndvi_p80",
    "wc_tree_frac", "wc_built_frac", "wc_bare_frac", "wc_water_frac",
]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Build the analysis frame: centred target + min-city filter + morphology."""
    work = df.copy()
    work["uhi_centred"] = within_city_center(work, target="uhi_anomaly_c", city="city_id").values
    # Drop cities with too few patches (< 30) to stabilise random intercept
    city_n = work.groupby("city_id").size()
    keep = city_n[city_n >= 30].index
    work = work[work.city_id.isin(keep)].copy()
    for c in MORPH_FEATS:
        if c not in work.columns:
            work[c] = 0.0
    # Coerce morphology features to float + fillna zeros
    for c in MORPH_FEATS:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    return work


def _fit_mixed(work: pd.DataFrame, features: list[str]):
    from statsmodels.regression.mixed_linear_model import MixedLM

    X = work[features].astype(float)
    # Clip extreme UHI outliers (likely cloud / fill pixels) to stabilise the fit.
    # Using the 1st–99th percentile of the centred target.
    y = work["uhi_centred"].values.astype(float)
    lo, hi = np.nanpercentile(y, [1, 99])
    mask = (y >= lo) & (y <= hi)
    y_clip = y[mask]
    X = X.loc[mask]
    groups = work["city_id"].astype(str).values[mask]
    # Standardise features (z-score) so β are comparable
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xz = (X - mu) / sd
    design_df = pd.concat([pd.Series(1.0, index=Xz.index, name="Intercept"), Xz], axis=1)
    md = MixedLM(endog=y_clip, exog=design_df.values, groups=groups)
    # Try lbfgs first; fall back to bfgs, powell, nm if it fails.
    fit = None
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            fit = md.fit(reml=True, method=method, maxiter=500)
            if fit.converged or method == "powell":
                break
        except Exception:
            continue
    if fit is None:
        raise RuntimeError("MixedLM fit failed with all optimizers")
    return fit, Xz, design_df, mu, sd


def _fit_reduced(work: pd.DataFrame):
    """Intercept-only MixedLM (same grouping) — baseline for partial R²."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    design = pd.DataFrame({"Intercept": np.ones(len(work))})
    md = MixedLM(endog=work["uhi_centred"].values, exog=design.values,
                 groups=work["city_id"].astype(str).values)
    return md.fit(reml=True, method="lbfgs", maxiter=500)


def _zone_mean_baseline(work: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import r2_score
    mean_by_zone = work.groupby("koppen_zone")["uhi_anomaly_c"].transform("mean")
    return {"r2_zone_mean_on_raw": float(r2_score(work["uhi_anomaly_c"], mean_by_zone))}


def _prediction_intervals(work: pd.DataFrame, features: list[str]) -> dict[str, float]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    X = work[features].values
    y = work["uhi_centred"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    qlo = GradientBoostingRegressor(loss="quantile", alpha=0.1,
                                    n_estimators=200, max_depth=4,
                                    random_state=42).fit(Xtr, ytr).predict(Xte)
    qhi = GradientBoostingRegressor(loss="quantile", alpha=0.9,
                                    n_estimators=200, max_depth=4,
                                    random_state=42).fit(Xtr, ytr).predict(Xte)
    return {
        "mean_80pi_width_c": float((qhi - qlo).mean()),
        "coverage_80pi": float(((yte >= qlo) & (yte <= qhi)).mean()),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    feats_path = Path(cfg.data_root) / "features" / f"{cfg.cities.name}.parquet"
    if not feats_path.exists():
        logger.error("Feature table not found at %s. Run run_features.py first.", feats_path)
        return
    df = read_patches(feats_path)
    df = quality_screen(df, abs_max_c=getattr(cfg, "uhi_abs_max_c", 40.0))
    work = _prepare(df)
    logger.info("Diagnostics: %d patches across %d cities, %d zones after prep",
                len(work), work.city_id.nunique(), work.koppen_zone.nunique())

    out_dir = Path.cwd()

    # ---- 1. Mixed-effects primary model ----
    logger.info("Fitting mixed-effects model ...")
    t0 = time.time()
    fit_full, Xz, design_full, mu, sd = _fit_mixed(work, MORPH_FEATS)
    logger.info("   fit in %.1fs", time.time() - t0)

    logger.info("Fitting reduced (intercept-only) model ...")
    fit_red = _fit_reduced(work)

    mc = marginal_conditional_r2(fit_full, Xz, work["uhi_centred"].values)
    icc = intraclass_correlation(fit_full)
    # Partial R² via marginal-R² difference
    mc_red = marginal_conditional_r2(fit_red, pd.DataFrame(np.zeros((len(work), 0))), work["uhi_centred"].values)
    partial = mc["marginal_r2"] - mc_red["marginal_r2"]

    # ---- Fixed-effect table with CI ----
    fe_table = {}
    ci = fit_full.conf_int()
    bse = fit_full.bse_fe
    tvals = fit_full.tvalues
    pvals = fit_full.pvalues
    fe = fit_full.fe_params
    # Normalise to numpy arrays regardless of statsmodels version
    fe_arr = fe.values if hasattr(fe, "values") else np.asarray(fe)
    bse_arr = bse.values if hasattr(bse, "values") else np.asarray(bse)
    tvals_arr = tvals.values if hasattr(tvals, "values") else np.asarray(tvals)
    pvals_arr = pvals.values if hasattr(pvals, "values") else np.asarray(pvals)
    ci_arr = ci.values if hasattr(ci, "values") else np.asarray(ci)
    exog_cols = list(design_full.columns)
    for i, name in enumerate(exog_cols):
        if name == "Intercept":
            continue
        beta_std = float(fe_arr[i])
        beta_raw = beta_std / float(sd[name]) if sd[name] > 0 else beta_std
        fe_table[name] = {
            "beta_standardised": beta_std,
            "beta_raw": beta_raw,
            "se": float(bse_arr[i]),
            "t": float(tvals_arr[i]),
            "p": float(pvals_arr[i]),
            "ci_low_std": float(ci_arr[i, 0]),
            "ci_high_std": float(ci_arr[i, 1]),
            "significant_95": bool(abs(tvals_arr[i]) > 1.96),
        }

    # ---- Per-zone marginal R² (aligned with the clipped fit sample) ----
    zones_aligned = work.loc[Xz.index, "koppen_zone"].values
    per_zone = per_zone_marginal_r2(fit_full, Xz, zones_aligned)

    # ---- Within-city LOCO R² (linear predictor) ----
    def _linear_fitter(X, y):
        from sklearn.linear_model import Ridge

        m = Ridge(alpha=1.0, random_state=42)
        m.fit(X, y)
        return m

    wc_loco = within_city_loco_r2(
        work,
        centred_target_col="uhi_centred",
        feature_cols=MORPH_FEATS,
        city_col="city_id",
        model_fitter=_linear_fitter,
    )

    # ---- Zone-mean baseline (on RAW uhi_anomaly_c) ----
    base = _zone_mean_baseline(work)

    # ---- Calibration (align to the clipped sample) ----
    fitted = np.asarray(fit_full.fittedvalues, dtype=float)
    y_clipped = work.loc[Xz.index, "uhi_centred"].values
    calib = calibration_slope(y_clipped, fitted)

    # ---- Prediction intervals ----
    pi = _prediction_intervals(work, MORPH_FEATS)

    # ---- Bootstrap SHAP stability (XGBoost on centred target) ----
    import shap
    from xgboost import XGBRegressor
    rng = np.random.default_rng(42)
    n_boot = 20
    top5_sets = []
    params = dict(max_depth=6, learning_rate=0.05, n_estimators=500,
                   subsample=0.85, random_state=42, n_jobs=-1, verbosity=0)
    feat_counts = {c: 0 for c in MORPH_FEATS}
    X_full = work[MORPH_FEATS].astype(float).values
    y_full = work["uhi_centred"].values
    for _ in range(n_boot):
        idx = rng.choice(len(X_full), size=len(X_full), replace=True)
        m = XGBRegressor(**params, eval_metric="rmse")
        m.fit(X_full[idx], y_full[idx], verbose=False)
        sv = shap.TreeExplainer(m).shap_values(X_full[idx])
        order = np.argsort(-np.abs(sv).mean(axis=0))[:5]
        top = {MORPH_FEATS[i] for i in order}
        top5_sets.append(top)
        for f in top:
            feat_counts[f] += 1
    jaccards = []
    for i in range(len(top5_sets)):
        for j in range(i + 1, len(top5_sets)):
            a, b = top5_sets[i], top5_sets[j]
            jaccards.append(len(a & b) / len(a | b) if (a | b) else 0.0)
    boot = {
        "shap_topk_mean_jaccard": float(np.mean(jaccards)),
        "feature_frequency_in_top5": dict(sorted(feat_counts.items(), key=lambda x: -x[1])),
        "n_boot": n_boot,
    }

    # ---- Consolidate report ----
    report = {
        "n_patches": len(work),
        "n_cities": int(work.city_id.nunique()),
        "n_zones": int(work.koppen_zone.nunique()),
        "mixed_effects": {
            "marginal_r2": mc["marginal_r2"],
            "conditional_r2": mc["conditional_r2"],
            "icc": icc,
            "partial_r2_over_intercept_only": partial,
            "per_zone_marginal_r2": per_zone,
            "fixed_effects": fe_table,
            "n_significant_features_95": sum(1 for v in fe_table.values() if v["significant_95"]),
        },
        "within_city_loco": wc_loco,
        "zone_mean_baseline": base,
        "calibration_slope": calib,
        "prediction_intervals_80": pi,
        "bootstrap_shap_stability": boot,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(report, indent=2))
    logger.info("=== Diagnostics summary ===")
    logger.info("   marginal R² = %.3f   conditional R² = %.3f   ICC = %.3f",
                mc["marginal_r2"], mc["conditional_r2"], icc)
    logger.info("   partial R² (morphology over city-only) = %.3f", partial)
    logger.info("   within-city LOCO R² (linear) = %.3f", wc_loco["r2"])
    logger.info("   zone-mean baseline R² (RAW) = %.3f", base["r2_zone_mean_on_raw"])
    logger.info("   calibration slope = %.3f", calib)
    logger.info("   bootstrap SHAP Jaccard (top-5) = %.3f", boot["shap_topk_mean_jaccard"])
    logger.info("   significant features (95%%) = %d/%d",
                report["mixed_effects"]["n_significant_features_95"], len(MORPH_FEATS))
    logger.info("   full report: %s", out_dir / "diagnostics.json")


if __name__ == "__main__":
    main()
