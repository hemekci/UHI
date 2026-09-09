"""Stage-3 step 3: train XGBoost UHI model(s) with leave-one-city-out CV and SHAP.

Usage:
    python run/pipeline/run_model.py cities=demo model=xgboost_global
    python run/pipeline/run_model.py cities=demo model=xgboost_perzone
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.model import ModelFactory  # noqa: E402
from src.utils import get_logger, read_patches, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


_NUMERIC_FEATURES = [
    "building_density",
    "height_mean",
    "height_cv",
    "ndvi_p80",
    "wc_tree_frac",
    "wc_grass_frac",
    "wc_cropland_frac",
    "wc_built_frac",
    "wc_bare_frac",
    "wc_water_frac",
    "impervious_frac",
    "vegetation_frac",
    "elevation_m",
    "era5_summer_t2m_c",
]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    data_path = Path(cfg.data_root) / "features" / f"{cfg.cities.name}.parquet"
    if not data_path.exists():
        logger.error("Feature table not found at %s. Run run_features.py first.", data_path)
        return
    df = read_patches(data_path)
    logger.info("Loaded %d patches", len(df))

    feats = [c for c in _NUMERIC_FEATURES if c in df.columns]
    X = df[feats].fillna(0.0).copy()
    if cfg.model.per_zone:
        X["koppen_zone"] = df["koppen_zone"].astype(str)
    y = df["uhi_anomaly_c"].astype(float)
    cities = df["city_id"].astype(str)

    # Leave-one-city-out CV: predict each city with a model trained on the others.
    logger.info("Leave-one-city-out CV on %d cities", cities.nunique())
    preds = np.full(len(df), np.nan)
    for city_id in cities.unique():
        val_mask = cities == city_id
        X_train = X.loc[~val_mask]
        y_train = y.loc[~val_mask]
        X_val = X.loc[val_mask]
        y_val = y.loc[val_mask]
        model = ModelFactory(cfg.model)
        # Simple internal split for early stopping: last 20% of train for val.
        cut = int(len(X_train) * 0.8)
        X_tr, X_vl = X_train.iloc[:cut], X_train.iloc[cut:]
        y_tr, y_vl = y_train.iloc[:cut], y_train.iloc[cut:]
        try:
            model.fit(X_tr, y_tr, X_vl, y_vl)
            p = model.predict(X_val)
            preds[val_mask.values] = p
            logger.info(
                "  fold held-out city=%s n_val=%d rmse=%.2f",
                city_id, int(val_mask.sum()), float(np.sqrt(np.mean((p - y_val.values) ** 2))),
            )
        except Exception:
            logger.exception("  fold city=%s failed", city_id)

    valid = ~np.isnan(preds)
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y.loc[valid], preds[valid])))
    r2 = float(r2_score(y.loc[valid], preds[valid]))
    logger.info("Out-of-sample R^2=%.3f RMSE=%.3f on %d/%d patches", r2, rmse, int(valid.sum()), len(df))

    # Also fit a final model on ALL data and compute SHAP.
    final = ModelFactory(cfg.model)
    cut = int(len(X) * 0.85)
    final.fit(X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:])
    report = final.report(X.iloc[cut:], y.iloc[cut:])
    logger.info("Final model SHAP top-5: %s", report.shap_top_features)

    out_dir = Path.cwd()
    summary = {
        "model": cfg.model.name,
        "per_zone": bool(cfg.model.per_zone),
        "n_patches": len(df),
        "n_cities": int(cities.nunique()),
        "features_used": feats + (["koppen_zone"] if cfg.model.per_zone else []),
        "oos_r2_loco": r2,
        "oos_rmse_loco": rmse,
        "final_shap_top_5": report.shap_top_features,
    }
    (out_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))

    # Also save per-patch predictions for downstream 2050 re-scoring.
    pred_df = df[["patch_id", "city_id", "koppen_zone", "uhi_anomaly_c", "building_density"]].copy()
    pred_df["uhi_pred_loco"] = preds
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    # Persist the final model.
    final.save(out_dir / "final_model")
    logger.info("Saved summary, predictions.parquet, final_model -> %s", out_dir)


if __name__ == "__main__":
    main()
