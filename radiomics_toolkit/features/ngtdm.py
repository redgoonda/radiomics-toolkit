"""Neighborhood Gray-Tone Difference Matrix (NGTDM) features — 5 features."""

from __future__ import annotations

import numpy as np

from .base import FeatureExtractor


class NGTDMExtractor(FeatureExtractor):
    """Compute 5 NGTDM features (Amadasun & King, 1989)."""

    prefix = "ngtdm"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        img = image.astype(np.float64)
        n_levels = int(img[mask > 0].max()) if np.any(mask > 0) else 1

        s_i, n_i = _build_ngtdm(img, mask, n_levels)
        feats = _ngtdm_features(s_i, n_i, n_levels)
        return self._prefixed(feats)


# ---------------------------------------------------------------------------
# NGTDM construction
# ---------------------------------------------------------------------------

def _build_ngtdm(
    image: np.ndarray, mask: np.ndarray, n_levels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build NGTDM arrays s_i (sum of absolute differences) and n_i (counts).

    For each voxel i in the mask with grey level g_i, compute
    A_i = average grey level of its 26-connected (3D) / 8-connected (2D) neighbours.
    Then s[g_i] += |g_i - A_i|, n[g_i] += 1.
    """
    ndim = image.ndim
    shape = image.shape

    # Build neighbour offsets (all combinations of -1,0,+1 except origin)
    if ndim == 2:
        offsets = [
            (di, dj)
            for di in range(-1, 2)
            for dj in range(-1, 2)
            if (di, dj) != (0, 0)
        ]
    else:
        offsets = [
            (dz, dy, dx)
            for dz in range(-1, 2)
            for dy in range(-1, 2)
            for dx in range(-1, 2)
            if (dz, dy, dx) != (0, 0, 0)
        ]

    s_i = np.zeros(n_levels + 1, dtype=np.float64)
    n_i = np.zeros(n_levels + 1, dtype=np.int64)

    roi_coords = list(zip(*np.where(mask > 0)))

    for coord in roi_coords:
        gl = int(image[coord])
        if gl < 1 or gl > n_levels:
            continue

        neighbour_vals = []
        for offset in offsets:
            nc = tuple(c + o for c, o in zip(coord, offset))
            if all(0 <= nc[d] < shape[d] for d in range(ndim)) and mask[nc] > 0:
                neighbour_vals.append(float(image[nc]))

        if not neighbour_vals:
            continue

        avg_neighbour = float(np.mean(neighbour_vals))
        s_i[gl] += abs(gl - avg_neighbour)
        n_i[gl] += 1

    return s_i[1:], n_i[1:]  # drop index 0, return 1-based as 0-indexed arrays


# ---------------------------------------------------------------------------
# NGTDM features
# ---------------------------------------------------------------------------

def _ngtdm_features(
    s_i: np.ndarray, n_i: np.ndarray, n_levels: int
) -> dict[str, float]:
    """Compute the 5 NGTDM features."""
    N = n_i.sum()
    if N == 0:
        return {k: float("nan") for k in ["coarseness", "contrast", "busyness", "complexity", "strength"]}

    # Normalised probabilities
    p_i = n_i / N  # shape (n_levels,)
    levels = np.arange(1, n_levels + 1)

    # Coarseness
    denom_coarse = float(np.sum(p_i * s_i))
    coarseness = 1.0 / denom_coarse if denom_coarse > 1e-12 else float("nan")

    # Contrast: (1/(Ng*(Ng-1))) * sum_ij(p_i * p_j * (i-j)^2) * (1/N) * sum_i(s_i)
    Ng = int(np.sum(n_i > 0))
    if Ng > 1:
        i_idx = levels.reshape(-1, 1)
        j_idx = levels.reshape(1, -1)
        p_outer = np.outer(p_i, p_i)
        contrast = float(
            (1.0 / (Ng * (Ng - 1))) * np.sum(p_outer * (i_idx - j_idx) ** 2) *
            (1.0 / N) * np.sum(s_i)
        )
    else:
        contrast = 0.0

    # Busyness
    num_busy = float(np.sum(p_i * s_i))  # reuse from coarseness
    denom_busy_vals = []
    for idx_i, (pi, si) in enumerate(zip(p_i, s_i)):
        for idx_j, (pj, sj) in enumerate(zip(p_i, s_i)):
            if pi > 0 and pj > 0:
                denom_busy_vals.append(abs((idx_i + 1) * pi - (idx_j + 1) * pj))
    denom_busy = sum(denom_busy_vals)
    busyness = float(num_busy / denom_busy) if denom_busy > 1e-12 else float("nan")

    # Complexity
    complexity_sum = 0.0
    for idx_i, (pi, si) in enumerate(zip(p_i, s_i)):
        for idx_j, (pj, sj) in enumerate(zip(p_i, s_i)):
            if pi > 0 and pj > 0:
                gl_diff = abs((idx_i + 1) - (idx_j + 1))
                denom_c = pi + pj
                if denom_c > 0:
                    complexity_sum += gl_diff * (pi * si + pj * sj) / denom_c
    complexity = float(complexity_sum / N) if N > 0 else float("nan")

    # Strength
    strength_num = 0.0
    strength_denom = float(np.sum(s_i))
    for idx_i, (pi, si) in enumerate(zip(p_i, s_i)):
        for idx_j, (pj, sj) in enumerate(zip(p_i, s_i)):
            if pi > 0 and pj > 0:
                strength_num += (pi + pj) * ((idx_i + 1) - (idx_j + 1)) ** 2
    strength = float(strength_num / strength_denom) if strength_denom > 1e-12 else float("nan")

    return {
        "coarseness": coarseness,
        "contrast": contrast,
        "busyness": busyness,
        "complexity": complexity,
        "strength": strength,
    }
