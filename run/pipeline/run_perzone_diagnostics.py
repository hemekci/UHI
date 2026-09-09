"""Per-Köppen-zone mixed-effects diagnostics — random slopes analogue.

For each Köppen zone with at least `min_patches` patches across `min_cities`
cities, fit a separate linear mixed-effects model on the within-city-centred
UHI target with city random intercept. Reports per-zone marginal R²,
per-zone β coefficients with 95 % CI, and pooled summary across zones.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import quality_screen, within_city_center  # noqa: E402
from src.metrics import intraclass_correlation, marginal_conditional_r2  # noqa: E402
from src.utils import get_logger, read_patches, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)

MORPH_FEATS = [
    "building_density", "height_mean", "height_cv",
    "ndvi_p80",
    "wc_tree_frac", "wc_built_frac", "wc_bare_frac", "wc_water_frac",
]

MIN_PATCHES_PER_ZONE = 500
MIN_CITIES_PER_ZONE = 2


def _fit_zone(zone_df: pd.DataFrame):
    """Fit MixedLM on within-zone data; return (fit, Xz, column order, mu, sd)."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    y = zone_df["uhi_centred"].values.astype(float)
    lo, hi = np.nanpercentile(y, [1, 99])
    mask = (y >= lo) & (y <= hi) & np.isfinite(y)
    X = zone_df.loc[mask, MORPH_FEATS].astype(float).fillna(0.0)
    # Drop features with zero variance in this zone
    var = X.var()
    keep = list(var[var > 1e-8].index)
    X = X[keep]
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xz = ((X - mu) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_clip = y[mask]
    design = pd.concat(
        [pd.Series(1.0, index=Xz.index, name="Intercept"), Xz], axis=1
    )
    groups = zone_df.loc[mask, "city_id"].astype(str).values
    md = MixedLM(endog=y_clip, exog=design.values, groups=groups)
    for method in ("lbfgs", "bfgs", "powell"):
        try:
            fit = md.fit(reml=True, method=method, maxiter=500)
            if fit.converged or method == "powell":
                break
        except Exception:
            fit = None
            continue
    return fit, Xz, list(design.columns), mu, sd, keep


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    feats_path = Path(cfg.data_root) / "features" / f"{cfg.cities.name}.parquet"
    df = read_patches(feats_path)
    df = quality_screen(df, abs_max_c=getattr(cfg, "uhi_abs_max_c", 40.0))

    df["uhi_centred"] = within_city_center(df).values
    # min-city filter within zone
    city_n = df.groupby("city_id").size()
    df = df[df.city_id.isin(city_n[city_n >= 30].index)].copy()

    report: dict = {"per_zone": {}, "global_weighted": None}

    for zone in sorted(df.koppen_zone.unique()):
        sub = df[df.koppen_zone == zone].copy()
        if len(sub) < MIN_PATCHES_PER_ZONE:
            continue
        if sub.city_id.nunique() < MIN_CITIES_PER_ZONE:
            continue
        logger.info("Fitting zone %s (n=%d patches, %d cities)",
                    zone, len(sub), sub.city_id.nunique())
        try:
            fit, Xz, cols, mu, sd, keep_feats = _fit_zone(sub)
        except Exception as e:
            logger.warning("   zone %s failed: %s", zone, e)
            continue
        if fit is None:
            logger.warning("   zone %s did not converge; skipping", zone)
            continue
        mc = marginal_conditional_r2(fit, Xz, sub["uhi_centred"].values)
        icc = intraclass_correlation(fit)

        # Extract coefficients
        fe = fit.fe_params
        fe_arr = fe.values if hasattr(fe, "values") else np.asarray(fe)
        bse = fit.bse_fe
        bse_arr = bse.values if hasattr(bse, "values") else np.asarray(bse)
        tvals = fit.tvalues
        tvals_arr = tvals.values if hasattr(tvals, "values") else np.asarray(tvals)
        pvals = fit.pvalues
        pvals_arr = pvals.values if hasattr(pvals, "values") else np.asarray(pvals)
        ci = fit.conf_int()
        ci_arr = ci.values if hasattr(ci, "values") else np.asarray(ci)
        fe_table = {}
        for i, name in enumerate(cols):
            if name == "Intercept":
                continue
            fe_table[name] = {
                "beta_std": float(fe_arr[i]),
                "se": float(bse_arr[i]),
                "t": float(tvals_arr[i]),
                "p": float(pvals_arr[i]),
                "ci_low": float(ci_arr[i, 0]),
                "ci_high": float(ci_arr[i, 1]),
                "significant_95": bool(abs(tvals_arr[i]) > 1.96),
            }

        report["per_zone"][zone] = {
            "n_patches": len(sub),
            "n_cities": int(sub.city_id.nunique()),
            "marginal_r2": mc["marginal_r2"],
            "conditional_r2": mc["conditional_r2"],
            "icc": icc,
            "n_significant": sum(1 for v in fe_table.values() if v["significant_95"]),
            "features_used": keep_feats,
            "fixed_effects": fe_table,
        }
        logger.info("   %s: marginal R² = %.3f  conditional R² = %.3f  ICC = %.3f  sig = %d/%d",
                    zone, mc["marginal_r2"], mc["conditional_r2"], icc,
                    report["per_zone"][zone]["n_significant"], len(fe_table))

    # Weighted pooled marginal R²
    total_n = sum(v["n_patches"] for v in report["per_zone"].values())
    if total_n > 0:
        weighted = sum(v["marginal_r2"] * v["n_patches"] for v in report["per_zone"].values()) / total_n
        report["global_weighted"] = {
            "marginal_r2_weighted": weighted,
            "n_zones_included": len(report["per_zone"]),
            "total_patches": total_n,
        }

    out = Path.cwd() / "diagnostics_perzone.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out)
    if report["global_weighted"]:
        logger.info("=== Pooled (weighted by zone n) marginal R² = %.3f over %d zones, %d patches ===",
                    report["global_weighted"]["marginal_r2_weighted"],
                    report["global_weighted"]["n_zones_included"],
                    report["global_weighted"]["total_patches"])


if __name__ == "__main__":
    main()
