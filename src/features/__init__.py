"""Per-patch feature engineering: continuous morphology + target + centring."""

from .centering import within_city_center, within_city_zscore
from .morphology_features import (
    building_density,
    canyon_aspect_proxy,
    height_mean_cv,
    orientation_entropy,
)
from .screening import DEFAULT_UHI_ABS_MAX_C, quality_screen
from .target import uhi_anomaly

__all__ = [
    "DEFAULT_UHI_ABS_MAX_C",
    "building_density",
    "canyon_aspect_proxy",
    "height_mean_cv",
    "orientation_entropy",
    "quality_screen",
    "uhi_anomaly",
    "within_city_center",
    "within_city_zscore",
]
