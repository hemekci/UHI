"""Single global XGBoost model (koppen_zone is a feature)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUHIModel, ModelReport
from .factory import register_model


@register_model("xgboost_global")
class XGBoostGlobalModel(BaseUHIModel):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._model = None
        self._explainer = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        from xgboost import XGBRegressor

        self._feature_names = list(X_train.columns)
        params = dict(self.cfg.params)
        early = params.pop("early_stopping_rounds", None)
        self._model = XGBRegressor(**params, eval_metric="rmse")
        self._model.fit(
            X_train.values,
            y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        assert self._model is not None, "Model is not fitted"
        return self._model.predict(X[self._feature_names].values)

    def shap_values(self, X: pd.DataFrame) -> NDArray[np.floating]:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer.shap_values(X[self._feature_names].values)

    def report(
        self, X_holdout: pd.DataFrame, y_holdout: pd.Series
    ) -> ModelReport:
        from sklearn.metrics import mean_squared_error, r2_score

        y_pred = self.predict(X_holdout)
        r2 = float(r2_score(y_holdout, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred)))

        sv = self.shap_values(X_holdout)
        mean_abs = np.abs(sv).mean(axis=0)
        top_k = self.cfg.shap.top_k_features
        order = np.argsort(-mean_abs)[:top_k]
        assert self._feature_names is not None
        top = [(self._feature_names[i], float(mean_abs[i])) for i in order]

        return ModelReport(
            r2_holdout=r2,
            rmse_holdout_c=rmse,
            shap_top_features=top,
            n_train=0,
            n_holdout=len(X_holdout),
            zone=None,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert self._model is not None
        self._model.save_model(str(path / "xgb_global.json"))

    def load(self, path: Path) -> None:
        from xgboost import XGBRegressor

        self._model = XGBRegressor()
        self._model.load_model(str(path / "xgb_global.json"))
