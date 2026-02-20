"""Gray-Level Size-Zone Matrix (GLSZM) features — 16 features.

The GLSZM counts connected zones (connected components) of the same grey level.
Uses 26-connectivity in 3D, 8-connectivity in 2D.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as nd_label

from .base import FeatureExtractor


class GLSZMExtractor(FeatureExtractor):
    """Compute 16 GLSZM features."""

    prefix = "glszm"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        img = image.astype(np.int32)
        n_levels = int(img[mask > 0].max()) if np.any(mask > 0) else 1

        glszm = _build_glszm(img, mask, n_levels)
        feats = _glszm_features(glszm, n_levels)
        return self._prefixed(feats)


# ---------------------------------------------------------------------------
# GLSZM construction
# ---------------------------------------------------------------------------

def _build_glszm(image: np.ndarray, mask: np.ndarray, n_levels: int) -> np.ndarray:
    """Build a GLSZM.

    Returns array of shape (n_levels, max_zone_size).
    """
    ndim = image.ndim
    structure = np.ones([3] * ndim, dtype=int)  # full connectivity

    zone_sizes: list[tuple[int, int]] = []  # (grey_level 0-based, zone_size)

    for gl in range(1, n_levels + 1):
        gl_mask = (image == gl) & (mask > 0)
        if not gl_mask.any():
            continue
        labeled, n_zones = nd_label(gl_mask, structure=structure)
        for zone_id in range(1, n_zones + 1):
            size = int((labeled == zone_id).sum())
            zone_sizes.append((gl - 1, size))

    if not zone_sizes:
        return np.zeros((n_levels, 1), dtype=np.float64)

    max_size = max(s for _, s in zone_sizes)
    glszm = np.zeros((n_levels, max_size), dtype=np.float64)
    for gl_0, sz in zone_sizes:
        if 0 <= gl_0 < n_levels:
            glszm[gl_0, sz - 1] += 1

    return glszm


# ---------------------------------------------------------------------------
# GLSZM features
# ---------------------------------------------------------------------------

_GLSZM_FEATURE_NAMES = [
    "small_area_emphasis", "large_area_emphasis",
    "gray_level_non_uniformity", "gray_level_non_uniformity_normalized",
    "size_zone_non_uniformity", "size_zone_non_uniformity_normalized",
    "zone_percentage", "low_gray_level_zone_emphasis", "high_gray_level_zone_emphasis",
    "small_area_low_gray_level_emphasis", "small_area_high_gray_level_emphasis",
    "large_area_low_gray_level_emphasis", "large_area_high_gray_level_emphasis",
    "gray_level_variance", "zone_variance", "zone_entropy",
]


def _glszm_features(glszm: np.ndarray, n_levels: int) -> dict[str, float]:
    Ng, Ns = glszm.shape
    total = glszm.sum()

    if total == 0:
        return {k: float("nan") for k in _GLSZM_FEATURE_NAMES}

    i = np.arange(1, Ng + 1).reshape(-1, 1)  # grey levels
    j = np.arange(1, Ns + 1).reshape(1, -1)  # zone sizes

    p = glszm / total

    ri = glszm.sum(axis=1)  # zones per grey level
    rj = glszm.sum(axis=0)  # zones per zone size

    small_area = float(np.sum(glszm / j ** 2) / total)
    large_area = float(np.sum(glszm * j ** 2) / total)
    glnu = float(np.sum(ri ** 2) / total)
    glnu_norm = float(np.sum(ri ** 2) / total ** 2)
    sznu = float(np.sum(rj ** 2) / total)
    sznu_norm = float(np.sum(rj ** 2) / total ** 2)

    # zone_percentage = total_zones / total_voxels_in_roi (approx)
    total_voxels = float(np.sum(j * glszm))
    zone_pct = float(total / total_voxels) if total_voxels > 0 else float("nan")

    low_gl = float(np.sum(glszm / i ** 2) / total)
    high_gl = float(np.sum(glszm * i ** 2) / total)

    small_low = float(np.sum(glszm / (i ** 2 * j ** 2)) / total)
    small_high = float(np.sum(glszm * i ** 2 / j ** 2) / total)
    large_low = float(np.sum(glszm * j ** 2 / i ** 2) / total)
    large_high = float(np.sum(glszm * i ** 2 * j ** 2) / total)

    mu_i = float(np.sum(p * i))
    gl_var = float(np.sum(p * (i - mu_i) ** 2))

    mu_j = float(np.sum(p * j))
    zone_var = float(np.sum(p * (j - mu_j) ** 2))

    eps = 1e-12
    zone_entropy = float(-np.sum(p * np.log2(p + eps)))

    return {
        "small_area_emphasis": small_area,
        "large_area_emphasis": large_area,
        "gray_level_non_uniformity": glnu,
        "gray_level_non_uniformity_normalized": glnu_norm,
        "size_zone_non_uniformity": sznu,
        "size_zone_non_uniformity_normalized": sznu_norm,
        "zone_percentage": zone_pct,
        "low_gray_level_zone_emphasis": low_gl,
        "high_gray_level_zone_emphasis": high_gl,
        "small_area_low_gray_level_emphasis": small_low,
        "small_area_high_gray_level_emphasis": small_high,
        "large_area_low_gray_level_emphasis": large_low,
        "large_area_high_gray_level_emphasis": large_high,
        "gray_level_variance": gl_var,
        "zone_variance": zone_var,
        "zone_entropy": zone_entropy,
    }
