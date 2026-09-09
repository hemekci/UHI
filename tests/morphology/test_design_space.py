"""Tests for the morphology feature-schema classes."""

import numpy as np
import pytest

from src.morphology import DesignSpace, DesignVariable


@pytest.fixture
def design_space() -> DesignSpace:
    variables = tuple(
        DesignVariable(name=f"v{i}", lower=0.0, upper=1.0, unit="ratio") for i in range(5)
    )
    return DesignSpace(variables=variables)


def test_design_variable_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        DesignVariable(name="bad", lower=1.0, upper=0.5, unit="ratio")


def test_design_space_properties(design_space):
    assert design_space.n_variables == 5
    assert design_space.names == ("v0", "v1", "v2", "v3", "v4")
    assert np.allclose(design_space.lower_bounds, 0.0)
    assert np.allclose(design_space.upper_bounds, 1.0)


def test_denormalize_inverts_normalize(design_space):
    rng = np.random.default_rng(0)
    x = rng.uniform(design_space.lower_bounds, design_space.upper_bounds, size=(10, 5))
    recovered = design_space.denormalize(design_space.normalize(x))
    np.testing.assert_allclose(x, recovered, atol=1e-10)
