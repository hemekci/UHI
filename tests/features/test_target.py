"""Target-variable smoke tests."""

import numpy as np

from src.features import uhi_anomaly


def test_uhi_anomaly_scalar():
    assert uhi_anomaly(33.0, 30.0) == 3.0


def test_uhi_anomaly_array():
    arr = np.array([31.0, 32.0, 33.0])
    out = uhi_anomaly(arr, 30.0)
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0]))
