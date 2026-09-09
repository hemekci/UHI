"""Cluster a Pareto front into ≤5 designer-consumable archetypes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TypologyClustering:
    k: int
    labels: NDArray[np.integer]
    centroids_parameter: NDArray[np.floating]
    centroids_objective: NDArray[np.floating]
    silhouette: float


def cluster_pareto_front(
    X_pareto: NDArray[np.floating],
    F_pareto: NDArray[np.floating],
    k_candidates: Sequence[int] = (3, 4, 5),
    random_seed: int = 42,
) -> TypologyClustering:
    """Standardize morphology parameters, run k-means for each k candidate,
    and pick the k with the highest silhouette score.

    Clustering is done in *parameter* space so the resulting archetypes correspond
    to designable morphology configurations, not to objective-space regions.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if len(X_pareto) < max(k_candidates) + 1:
        raise ValueError(
            f"Not enough Pareto points ({len(X_pareto)}) to cluster into up to {max(k_candidates)} groups"
        )

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_pareto)

    best: TypologyClustering | None = None
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = km.fit_predict(X_std)
        sil = float(silhouette_score(X_std, labels))
        centroids_param = np.array(
            [X_pareto[labels == i].mean(axis=0) for i in range(k)]
        )
        centroids_obj = np.array(
            [F_pareto[labels == i].mean(axis=0) for i in range(k)]
        )
        candidate = TypologyClustering(
            k=k,
            labels=labels,
            centroids_parameter=centroids_param,
            centroids_objective=centroids_obj,
            silhouette=sil,
        )
        if best is None or sil > best.silhouette:
            best = candidate

    assert best is not None
    return best
