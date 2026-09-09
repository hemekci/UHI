"""Tests for Pareto-front clustering."""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for clustering")

from src.typology import cluster_pareto_front


def test_cluster_pareto_front_returns_expected_shapes():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.1, size=(15, 5)),
        rng.normal(loc=3.0, scale=0.1, size=(15, 5)),
        rng.normal(loc=6.0, scale=0.1, size=(15, 5)),
    ])
    F = rng.normal(size=(45, 4))

    tc = cluster_pareto_front(X, F, k_candidates=(3, 4), random_seed=0)

    assert tc.k in (3, 4)
    assert tc.labels.shape == (45,)
    assert tc.centroids_parameter.shape == (tc.k, 5)
    assert tc.centroids_objective.shape == (tc.k, 4)
    assert tc.silhouette > 0  # well-separated synthetic clusters
