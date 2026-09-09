"""Target variable construction — patch UHI anomaly from summer-mean LST.

UHI anomaly is computed per Landsat scene, but here we only hold the aggregated
summer-mean LST per patch and the rural reference LST percentile for the same
city/scene set. The reduction across scenes (3-summer rolling mean) happens in
the ingest stage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def uhi_anomaly(
    patch_lst_c: NDArray[np.floating] | float,
    rural_reference_lst_c: float,
) -> NDArray[np.floating] | float:
    """UHI anomaly = patch summer-mean LST − rural reference LST P50.

    Inputs are already in Celsius. This is the supervised-learning target.
    """
    return np.asarray(patch_lst_c, dtype=float) - float(rural_reference_lst_c)
