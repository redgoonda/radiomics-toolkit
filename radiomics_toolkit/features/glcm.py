"""Gray-Level Co-occurrence Matrix (GLCM) features.

Directions: 4 (2D) or 13 (3D); results averaged across all directions.
"""

from __future__ import annotations

import numpy as np

from .base import FeatureExtractor
from ..utils import get_directions_2d, get_directions_3d


class GLCMExtractor(FeatureExtractor):
    """Compute 13 GLCM features averaged over all directions."""

    prefix = "glcm"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        # image should already be discretized (integer grey levels 1…N)
        img = image.astype(np.int32)
        n_levels = int(img[mask > 0].max()) if np.any(mask > 0) else 1

        directions = get_directions_2d() if img.ndim == 2 else get_directions_3d()

        feature_sums: dict[str, float] = {}
        count = 0

        for direction in directions:
            glcm = _build_glcm(img, mask, direction, n_levels)
            feats = _glcm_features(glcm)
            for k, v in feats.items():
                feature_sums[k] = feature_sums.get(k, 0.0) + v
            count += 1

        if count == 0:
            nan = float("nan")
            return self._prefixed({k: nan for k in _GLCM_FEATURE_NAMES})

        averaged = {k: v / count for k, v in feature_sums.items()}
        return self._prefixed(averaged)


# ---------------------------------------------------------------------------
# GLCM construction
# ---------------------------------------------------------------------------

def _build_glcm(
    image: np.ndarray,
    mask: np.ndarray,
    direction: tuple,
    n_levels: int,
) -> np.ndarray:
    """Build a normalised symmetric GLCM for one direction."""
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)

    # Shift indices
    shape = image.shape
    ndim = image.ndim

    if ndim == 2:
        di, dj = direction
        rows, cols = np.where(mask > 0)
        ni = rows + di
        nj = cols + dj
        in_bounds = (ni >= 0) & (ni < shape[0]) & (nj >= 0) & (nj < shape[1])
        # Two-step filter: bounds first, then mask membership
        valid = np.zeros(len(rows), dtype=bool)
        if in_bounds.any():
            valid[in_bounds] = mask[ni[in_bounds], nj[in_bounds]] > 0

        i_vals = image[rows[valid], cols[valid]] - 1
        j_vals = image[ni[valid], nj[valid]] - 1

        np.add.at(glcm, (i_vals, j_vals), 1)
        np.add.at(glcm, (j_vals, i_vals), 1)  # symmetric

    else:  # 3D
        dz, dy, dx = direction
        zz, yy, xx = np.where(mask > 0)
        nz = zz + dz
        ny = yy + dy
        nx = xx + dx
        in_bounds = (
            (nz >= 0) & (nz < shape[0]) &
            (ny >= 0) & (ny < shape[1]) &
            (nx >= 0) & (nx < shape[2])
        )
        nz, ny, nx = nz[in_bounds], ny[in_bounds], nx[in_bounds]
        zz, yy, xx = zz[in_bounds], yy[in_bounds], xx[in_bounds]
        in_mask = mask[nz, ny, nx] > 0
        zz, yy, xx = zz[in_mask], yy[in_mask], xx[in_mask]
        nz, ny, nx = nz[in_mask], ny[in_mask], nx[in_mask]

        i_vals = image[zz, yy, xx] - 1
        j_vals = image[nz, ny, nx] - 1

        np.add.at(glcm, (i_vals, j_vals), 1)
        np.add.at(glcm, (j_vals, i_vals), 1)

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


# ---------------------------------------------------------------------------
# GLCM feature computation
# ---------------------------------------------------------------------------

_GLCM_FEATURE_NAMES = [
    "energy", "entropy", "contrast", "homogeneity", "correlation",
    "dissimilarity", "autocorrelation", "cluster_shade", "cluster_prominence",
    "joint_average", "joint_energy", "imc1", "imc2",
]


def _glcm_features(glcm: np.ndarray) -> dict[str, float]:
    n = glcm.shape[0]
    if n == 0 or glcm.sum() == 0:
        return {k: float("nan") for k in _GLCM_FEATURE_NAMES}

    i_idx, j_idx = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")

    energy = float(np.sum(glcm ** 2))
    joint_energy = energy

    p = glcm
    eps = 1e-12
    entropy = float(-np.sum(p * np.log2(p + eps)))
    contrast = float(np.sum(p * (i_idx - j_idx) ** 2))
    homogeneity = float(np.sum(p / (1 + (i_idx - j_idx) ** 2)))
    dissimilarity = float(np.sum(p * np.abs(i_idx - j_idx)))

    # Marginal probabilities
    px = p.sum(axis=1)  # shape (n,)
    py = p.sum(axis=0)
    mu_x = float(np.sum(np.arange(1, n + 1) * px))
    mu_y = float(np.sum(np.arange(1, n + 1) * py))
    sig_x = float(np.sqrt(np.sum((np.arange(1, n + 1) - mu_x) ** 2 * px)))
    sig_y = float(np.sqrt(np.sum((np.arange(1, n + 1) - mu_y) ** 2 * py)))

    autocorrelation = float(np.sum(p * i_idx * j_idx))
    joint_average = mu_x

    if sig_x > eps and sig_y > eps:
        correlation = float(
            (np.sum(p * i_idx * j_idx) - mu_x * mu_y) / (sig_x * sig_y)
        )
    else:
        correlation = float("nan")

    # Cluster shade and prominence
    mu = (mu_x + mu_y) / 2
    cluster_shade = float(np.sum(p * (i_idx + j_idx - 2 * mu) ** 3))
    cluster_prominence = float(np.sum(p * (i_idx + j_idx - 2 * mu) ** 4))

    # Information measures of correlation
    # HXY = entropy of p; HX, HY = marginal entropies
    hx = float(-np.sum(px * np.log2(px + eps)))
    hy = float(-np.sum(py * np.log2(py + eps)))
    hxy = entropy
    # HXY1 = -sum p(i,j) log2( px(i)*py(j) )
    px_outer = np.outer(px, py)
    hxy1 = float(-np.sum(p * np.log2(px_outer + eps)))
    # HXY2 = -sum px(i)*py(j) log2( px(i)*py(j) )
    hxy2 = float(-np.sum(px_outer * np.log2(px_outer + eps)))

    max_hxy = max(hx, hy)
    imc1 = float((hxy - hxy1) / max_hxy) if max_hxy > eps else float("nan")

    val_imc2 = 1 - np.exp(-2 * (hxy2 - hxy))
    imc2 = float(np.sqrt(max(0.0, val_imc2)))

    return {
        "energy": energy,
        "entropy": entropy,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "correlation": correlation,
        "dissimilarity": dissimilarity,
        "autocorrelation": autocorrelation,
        "cluster_shade": cluster_shade,
        "cluster_prominence": cluster_prominence,
        "joint_average": joint_average,
        "joint_energy": joint_energy,
        "imc1": imc1,
        "imc2": imc2,
    }
