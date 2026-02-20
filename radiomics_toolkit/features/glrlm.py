"""Gray-Level Run-Length Matrix (GLRLM) features — 16 features."""

from __future__ import annotations

import numpy as np

from .base import FeatureExtractor
from ..utils import get_directions_2d, get_directions_3d


class GLRLMExtractor(FeatureExtractor):
    """Compute 16 GLRLM features averaged over all directions."""

    prefix = "glrlm"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        img = image.astype(np.int32)
        n_levels = int(img[mask > 0].max()) if np.any(mask > 0) else 1

        directions = get_directions_2d() if img.ndim == 2 else get_directions_3d()

        feature_sums: dict[str, float] = {}
        count = 0

        for direction in directions:
            glrlm = _build_glrlm(img, mask, direction, n_levels)
            feats = _glrlm_features(glrlm, n_levels)
            for k, v in feats.items():
                feature_sums[k] = feature_sums.get(k, 0.0) + (v if np.isfinite(v) else 0.0)
            count += 1

        if count == 0:
            return self._prefixed({k: float("nan") for k in _GLRLM_FEATURE_NAMES})

        averaged = {k: v / count for k, v in feature_sums.items()}
        return self._prefixed(averaged)


# ---------------------------------------------------------------------------
# GLRLM construction
# ---------------------------------------------------------------------------

def _build_glrlm(
    image: np.ndarray,
    mask: np.ndarray,
    direction: tuple,
    n_levels: int,
) -> np.ndarray:
    """Build a GLRLM for one direction.

    Returns array of shape (n_levels, max_run_length).
    """
    shape = image.shape
    visited = np.zeros(shape, dtype=bool)
    runs: list[tuple[int, int]] = []  # (grey_level, run_length)

    coords = list(zip(*np.where(mask > 0)))

    # Reverse direction to find run starts
    rev = tuple(-d for d in direction)

    for coord in coords:
        coord = tuple(coord)
        # Check if this is the start of a run (predecessor not in mask/direction)
        pred = tuple(c + d for c, d in zip(coord, rev))
        is_start = not (
            all(0 <= p < s for p, s in zip(pred, shape)) and mask[pred] > 0
            and image[pred] == image[coord]
        )
        if not is_start:
            continue

        # Traverse the run
        run_len = 1
        cur = tuple(c + d for c, d in zip(coord, direction))
        while (
            all(0 <= c < s for c, s in zip(cur, shape))
            and mask[cur] > 0
            and image[cur] == image[coord]
        ):
            run_len += 1
            cur = tuple(c + d for c, d in zip(cur, direction))

        gl = int(image[coord]) - 1  # 0-based
        runs.append((gl, run_len))

    if not runs:
        return np.zeros((n_levels, 1), dtype=np.float64)

    max_run = max(r for _, r in runs)
    glrlm = np.zeros((n_levels, max_run), dtype=np.float64)
    for gl, rl in runs:
        if 0 <= gl < n_levels:
            glrlm[gl, rl - 1] += 1

    return glrlm


# ---------------------------------------------------------------------------
# GLRLM features
# ---------------------------------------------------------------------------

_GLRLM_FEATURE_NAMES = [
    "short_run_emphasis", "long_run_emphasis",
    "gray_level_non_uniformity", "gray_level_non_uniformity_normalized",
    "run_length_non_uniformity", "run_length_non_uniformity_normalized",
    "run_percentage", "low_gray_level_run_emphasis", "high_gray_level_run_emphasis",
    "short_run_low_gray_level_emphasis", "short_run_high_gray_level_emphasis",
    "long_run_low_gray_level_emphasis", "long_run_high_gray_level_emphasis",
    "run_variance", "run_entropy", "gray_level_variance",
]


def _glrlm_features(glrlm: np.ndarray, n_levels: int) -> dict[str, float]:
    Ng, Nr = glrlm.shape
    total = glrlm.sum()

    if total == 0:
        return {k: float("nan") for k in _GLRLM_FEATURE_NAMES}

    i = np.arange(1, Ng + 1).reshape(-1, 1)  # grey levels (1-based)
    j = np.arange(1, Nr + 1).reshape(1, -1)  # run lengths (1-based)

    p = glrlm / total  # normalised

    ri = glrlm.sum(axis=1)  # runs per grey level
    rj = glrlm.sum(axis=0)  # runs per run length

    short_run_emphasis = float(np.sum(glrlm / j ** 2) / total)
    long_run_emphasis = float(np.sum(glrlm * j ** 2) / total)
    glnu = float(np.sum(ri ** 2) / total)
    glnu_norm = float(np.sum(ri ** 2) / total ** 2)
    rlnu = float(np.sum(rj ** 2) / total)
    rlnu_norm = float(np.sum(rj ** 2) / total ** 2)

    # Number of voxels in mask (approximate)
    n_voxels = max(total, 1)
    run_pct = float(total / n_voxels)

    low_gl = float(np.sum(glrlm / i ** 2) / total)
    high_gl = float(np.sum(glrlm * i ** 2) / total)

    short_low = float(np.sum(glrlm / (i ** 2 * j ** 2)) / total)
    short_high = float(np.sum(glrlm * i ** 2 / j ** 2) / total)
    long_low = float(np.sum(glrlm * j ** 2 / i ** 2) / total)
    long_high = float(np.sum(glrlm * i ** 2 * j ** 2) / total)

    # Run variance
    mu_j = float(np.sum(p * j))
    run_var = float(np.sum(p * (j - mu_j) ** 2))

    # Run entropy
    eps = 1e-12
    run_entropy = float(-np.sum(p * np.log2(p + eps)))

    # Grey-level variance
    mu_i = float(np.sum(p * i))
    gl_var = float(np.sum(p * (i - mu_i) ** 2))

    return {
        "short_run_emphasis": short_run_emphasis,
        "long_run_emphasis": long_run_emphasis,
        "gray_level_non_uniformity": glnu,
        "gray_level_non_uniformity_normalized": glnu_norm,
        "run_length_non_uniformity": rlnu,
        "run_length_non_uniformity_normalized": rlnu_norm,
        "run_percentage": run_pct,
        "low_gray_level_run_emphasis": low_gl,
        "high_gray_level_run_emphasis": high_gl,
        "short_run_low_gray_level_emphasis": short_low,
        "short_run_high_gray_level_emphasis": short_high,
        "long_run_low_gray_level_emphasis": long_low,
        "long_run_high_gray_level_emphasis": long_high,
        "run_variance": run_var,
        "run_entropy": run_entropy,
        "gray_level_variance": gl_var,
    }
