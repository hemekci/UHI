"""Immutable morphology design-space definitions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DesignVariable:
    """A single morphology design variable with bounds and unit metadata."""

    name: str
    lower: float
    upper: float
    unit: str

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError(
                f"Variable {self.name!r}: upper ({self.upper}) must exceed lower ({self.lower})"
            )


@dataclass(frozen=True)
class DesignSpace:
    """Collection of morphology variables defining the optimization search space."""

    variables: tuple[DesignVariable, ...]

    @classmethod
    def from_config(cls, cfg_variables: Iterable[dict]) -> DesignSpace:
        vars_tuple = tuple(
            DesignVariable(
                name=v["name"], lower=float(v["lower"]), upper=float(v["upper"]), unit=v["unit"]
            )
            for v in cfg_variables
        )
        return cls(variables=vars_tuple)

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def lower_bounds(self) -> NDArray[np.floating]:
        return np.array([v.lower for v in self.variables], dtype=float)

    @property
    def upper_bounds(self) -> NDArray[np.floating]:
        return np.array([v.upper for v in self.variables], dtype=float)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def denormalize(self, x_unit: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map a [0, 1]^n sample to the design-variable ranges."""
        return self.lower_bounds + x_unit * (self.upper_bounds - self.lower_bounds)

    def normalize(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map a design-variable sample to [0, 1]^n."""
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
