"""Abstract base class for UHI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ModelReport:
    """Held-out performance + attribution summary for one trained model."""

    r2_holdout: float
    rmse_holdout_c: float
    shap_top_features: list[tuple[str, float]]
    n_train: int
    n_holdout: int
    zone: str | None

    # Mixed-effects extensions (optional; None for non-hierarchical models)
    marginal_r2: float | None = None
    conditional_r2: float | None = None
    icc: float | None = None
    fixed_effects: dict[str, dict[str, float]] | None = None
    per_zone_r2: dict[str, float] | None = None


class BaseUHIModel(ABC):
    """All UHI models consume a feature DataFrame and a target Series."""

    name: str = "base"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]: ...

    @abstractmethod
    def shap_values(self, X: pd.DataFrame) -> NDArray[np.floating]: ...

    @abstractmethod
    def report(
        self, X_holdout: pd.DataFrame, y_holdout: pd.Series
    ) -> ModelReport: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
