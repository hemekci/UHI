"""Complementary XGBoost model fit on the within-city-centred UHI target.

Used purely for non-linear SHAP attribution ranking. Features are the 10
morphology variables; the city-constant context features (era5, elevation,
latitude, Köppen) are intentionally dropped to prevent the model from
learning city identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUHIModel, ModelReport
from .factory import register_model


@register_model("within_city_xgboost")
class WithinCityXGBoostModel(BaseUHIModel):
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
        params.pop("early_stopping_rounds", None)
        self._model = XGBRegressor(**params, eval_metric="rmse")
        self._model.fit(
            X_train.values,
            y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        assert self._model is not None
        return self._model.predict(X[self._feature_names].values)

    def shap_values(self, X: pd.DataFrame) -> NDArray[np.floating]:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer.shap_values(X[self._feature_names].values)

    def shap_bootstrap_stability(
        self, X: pd.DataFrame, y: pd.Series, n_boot: int = 20, top_k: int = 5
    ) -> dict[str, Any]:
        """Resample-with-replacement; report mean pairwise Jaccard of top-K features."""
        import shap
        from xgboost import XGBRegressor

        rng = np.random.default_rng(42)
        top_k_sets: list[set[str]] = []
        params = dict(self.cfg.params)
        params.pop("early_stopping_rounds", None)
        feature_counts: dict[str, int] = {c: 0 for c in self._feature_names}
        for _ in range(n_boot):
            idx = rng.choice(len(X), size=len(X), replace=True)
            m = XGBRegressor(**params, eval_metric="rmse")
            m.fit(X.iloc[idx].values, y.iloc[idx].values, verbose=False)
            sv = shap.TreeExplainer(m).shap_values(X.iloc[idx].values)
            order = np.argsort(-np.abs(sv).mean(axis=0))[:top_k]
            top_set = {self._feature_names[i] for i in order}
            top_k_sets.append(top_set)
            for feat in top_set:
                feature_counts[feat] += 1
        # Pairwise Jaccard
        jaccards = []
        for i in range(len(top_k_sets)):
            for j in range(i + 1, len(top_k_sets)):
                a, b = top_k_sets[i], top_k_sets[j]
                jaccards.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        return {
            "mean_pairwise_jaccard": float(np.mean(jaccards)),
            "feature_frequency_in_topk": dict(
                sorted(feature_counts.items(), key=lambda x: -x[1])
            ),
            "n_boot": n_boot,
            "top_k": top_k,
        }

    def report(
        self, X_holdout: pd.DataFrame, y_holdout: pd.Series
    ) -> ModelReport:
        from sklearn.metrics import mean_squared_error, r2_score

        pred = self.predict(X_holdout)
        r2 = float(r2_score(y_holdout, pred))
        rmse = float(np.sqrt(mean_squared_error(y_holdout, pred)))

        sv = self.shap_values(X_holdout)
        mean_abs = np.abs(sv).mean(axis=0)
        top_k = self.cfg.shap.top_k_features
        order = np.argsort(-mean_abs)[:top_k]
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
        self._model.save_model(str(path / "within_city_xgb.json"))

    def load(self, path: Path) -> None:
        from xgboost import XGBRegressor

        self._model = XGBRegressor()
        self._model.load_model(str(path / "within_city_xgb.json"))
