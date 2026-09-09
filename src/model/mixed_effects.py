"""Linear mixed-effects UHI model (statsmodels.MixedLM) — primary inferential model.

Fits Δ_p = β_0 + β·morphology_p + γ_city + ε with city random intercept.
Reports fixed-effect coefficients, their 95 % CIs, and variance components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import BaseUHIModel, ModelReport
from .factory import register_model


@register_model("mixed_effects")
class MixedEffectsUHIModel(BaseUHIModel):
    """Linear mixed-effects model with city random intercept."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.fit_result = None
        self._feature_cols: list[str] | None = None
        self._city_col: str = getattr(cfg, "city_col", "city_id")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Fit MixedLM on pooled (X_train + X_val) — validation split ignored."""
        from statsmodels.regression.mixed_linear_model import MixedLM

        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)

        # Derive feature columns (exclude grouping)
        self._feature_cols = [c for c in X_all.columns if c != self._city_col]
        design = X_all[self._feature_cols].astype(float)
        design = design.assign(Intercept=1.0)[["Intercept", *self._feature_cols]]

        groups = X_all[self._city_col].astype(str).values
        md = MixedLM(endog=y_all.values, exog=design.values, groups=groups)
        self.fit_result = md.fit(
            reml=getattr(self.cfg, "reml", True),
            method=getattr(self.cfg, "method", "lbfgs"),
            maxiter=getattr(self.cfg, "max_iter", 500),
        )
        # Attach column names for later recovery
        self.fit_result.model.exog_names = ["Intercept", *self._feature_cols]

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        assert self.fit_result is not None
        design = X[self._feature_cols].astype(float)
        design = design.assign(Intercept=1.0)[["Intercept", *self._feature_cols]]
        beta = np.asarray(self.fit_result.fe_params.values, dtype=float)
        return design.values @ beta

    def shap_values(self, X: pd.DataFrame) -> NDArray[np.floating]:
        """Linear fixed-effect contributions (equivalent to SHAP for linear models)."""
        assert self.fit_result is not None
        x = X[self._feature_cols].astype(float).values
        beta = np.asarray(self.fit_result.fe_params.values[1:], dtype=float)  # drop intercept
        means = x.mean(axis=0)
        # Per-sample contribution of each feature (value * beta) centred at mean contribution
        return (x - means) * beta

    def report(
        self, X_holdout: pd.DataFrame, y_holdout: pd.Series
    ) -> ModelReport:
        from sklearn.metrics import mean_squared_error, r2_score

        from ..metrics import intraclass_correlation, marginal_conditional_r2

        assert self.fit_result is not None
        pred = self.predict(X_holdout)
        r2 = float(r2_score(y_holdout, pred))
        rmse = float(np.sqrt(mean_squared_error(y_holdout, pred)))

        # Fixed-effect summary
        fe = self.fit_result.fe_params
        ci = self.fit_result.conf_int()
        tvals = self.fit_result.tvalues
        pvals = self.fit_result.pvalues
        bse = self.fit_result.bse_fe
        fe_table = {}
        for name in fe.index:
            if name == "Intercept":
                continue
            fe_table[name] = {
                "beta": float(fe[name]),
                "se": float(bse[name]) if name in bse.index else float("nan"),
                "t": float(tvals[name]) if name in tvals.index else float("nan"),
                "p": float(pvals[name]) if name in pvals.index else float("nan"),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }

        # Variance components (R² on pooled y fitted values vs target)
        design = X_holdout[self._feature_cols].astype(float)
        r2_dict = marginal_conditional_r2(self.fit_result, design, y_holdout.values)
        icc = intraclass_correlation(self.fit_result)

        top = sorted(
            [(k, abs(v["beta"])) for k, v in fe_table.items()],
            key=lambda x: -x[1],
        )[: self.cfg.shap.top_k_features]

        return ModelReport(
            r2_holdout=r2,
            rmse_holdout_c=rmse,
            shap_top_features=top,
            n_train=int(self.fit_result.nobs),
            n_holdout=len(X_holdout),
            zone=None,
            marginal_r2=r2_dict["marginal_r2"],
            conditional_r2=r2_dict["conditional_r2"],
            icc=icc,
            fixed_effects=fe_table,
        )

    def save(self, path: Path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.fit_result, path / "mixed_effects.pkl")

    def load(self, path: Path) -> None:
        import joblib

        self.fit_result = joblib.load(path / "mixed_effects.pkl")
