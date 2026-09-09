"""Per-Köppen-zone XGBoost ensemble — one model per zone, SHAP per zone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUHIModel, ModelReport
from .factory import register_model


@register_model("xgboost_perzone")
class XGBoostPerZoneModel(BaseUHIModel):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.zone_models: dict[str, XGBoostPerZoneModel._ZoneFit] = {}
        self._feature_names: list[str] | None = None

    class _ZoneFit:
        def __init__(self, model, explainer) -> None:
            self.model = model
            self.explainer = explainer

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        from xgboost import XGBRegressor

        if "koppen_zone" not in X_train.columns:
            raise KeyError("Per-zone model requires 'koppen_zone' column in features")

        feat_cols = [c for c in X_train.columns if c != "koppen_zone"]
        self._feature_names = feat_cols

        for zone in sorted(X_train["koppen_zone"].unique()):
            zt_mask = X_train["koppen_zone"] == zone
            zv_mask = X_val["koppen_zone"] == zone
            if zv_mask.sum() < 20:
                continue
            params = dict(self.cfg.params)
            params.pop("early_stopping_rounds", None)
            model = XGBRegressor(**params, eval_metric="rmse")
            model.fit(
                X_train.loc[zt_mask, feat_cols].values,
                y_train[zt_mask].values,
                eval_set=[(X_val.loc[zv_mask, feat_cols].values, y_val[zv_mask].values)],
                verbose=False,
            )
            self.zone_models[zone] = self._ZoneFit(model, None)

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        assert self._feature_names is not None
        preds = np.full(len(X), np.nan)
        for zone, fit in self.zone_models.items():
            mask = X["koppen_zone"] == zone
            if not mask.any():
                continue
            preds[mask.values] = fit.model.predict(X.loc[mask, self._feature_names].values)
        return preds

    def shap_values(self, X: pd.DataFrame) -> NDArray[np.floating]:
        import shap

        assert self._feature_names is not None
        sv = np.full((len(X), len(self._feature_names)), np.nan)
        for zone, fit in self.zone_models.items():
            mask = X["koppen_zone"] == zone
            if not mask.any():
                continue
            if fit.explainer is None:
                fit.explainer = shap.TreeExplainer(fit.model)
            sv[mask.values, :] = fit.explainer.shap_values(
                X.loc[mask, self._feature_names].values
            )
        return sv

    def report(
        self, X_holdout: pd.DataFrame, y_holdout: pd.Series
    ) -> ModelReport:
        from sklearn.metrics import mean_squared_error, r2_score

        assert self._feature_names is not None
        preds = self.predict(X_holdout)
        valid = ~np.isnan(preds)
        r2 = float(r2_score(y_holdout[valid], preds[valid])) if valid.any() else float("nan")
        rmse = (
            float(np.sqrt(mean_squared_error(y_holdout[valid], preds[valid])))
            if valid.any() else float("nan")
        )
        sv = self.shap_values(X_holdout)
        mean_abs = np.nanmean(np.abs(sv), axis=0)
        top_k = self.cfg.shap.top_k_features
        order = np.argsort(-mean_abs)[:top_k]
        top = [(self._feature_names[i], float(mean_abs[i])) for i in order]

        return ModelReport(
            r2_holdout=r2,
            rmse_holdout_c=rmse,
            shap_top_features=top,
            n_train=0,
            n_holdout=len(X_holdout),
            zone="multi",
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for zone, fit in self.zone_models.items():
            fit.model.save_model(str(path / f"xgb_{zone}.json"))

    def load(self, path: Path) -> None:
        from xgboost import XGBRegressor

        self.zone_models = {}
        for model_path in path.glob("xgb_*.json"):
            zone = model_path.stem.replace("xgb_", "")
            model = XGBRegressor()
            model.load_model(str(model_path))
            self.zone_models[zone] = self._ZoneFit(model, None)
