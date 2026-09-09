"""Unit tests for deterministic morphology-feature builders."""

import math

import pytest

from src.features import (
    building_density,
    canyon_aspect_proxy,
    height_mean_cv,
    orientation_entropy,
)


def test_building_density_clamped_between_0_and_1():
    assert building_density(500.0, 1000.0) == 0.5
    assert building_density(0.0, 1000.0) == 0.0
    assert building_density(2000.0, 1000.0) == 1.0  # clamped
    assert building_density(500.0, 0.0) == 0.0


def test_height_mean_cv_unweighted():
    mean_h, cv = height_mean_cv([10.0, 20.0, 30.0])
    assert mean_h == pytest.approx(20.0)
    assert cv == pytest.approx(math.sqrt(200.0 / 3.0) / 20.0, rel=1e-6)


def test_height_mean_cv_empty():
    assert height_mean_cv([]) == (0.0, 0.0)


def test_canyon_aspect_proxy_basic():
    assert canyon_aspect_proxy(20.0, 10.0) == 2.0
    assert canyon_aspect_proxy(10.0, 0.0) == 0.0
    assert canyon_aspect_proxy(1000.0, 1.0) == 10.0  # capped


def test_orientation_entropy_uniform_higher_than_single_bearing():
    uniform = orientation_entropy([i * 5.0 for i in range(36)])
    single = orientation_entropy([90.0] * 36)
    assert uniform > single
