"""Grey-level discretization.

Supports two strategies:
- ``fixed_count``  (default): divide the value range into *n* equal-width bins.
- ``fixed_width``           : specify a fixed bin width; number of bins is inferred.
"""

from __future__ import annotations

import numpy as np


def discretize(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    strategy: str = "fixed_count",
    bin_count: int = 64,
    bin_width: float | None = None,
) -> np.ndarray:
    """Return a discretized copy of *image* (only within *mask*).

    Voxels outside the mask are set to 0. Bin indices are 1-based (1 … N)
    to match IBSI convention.

    Parameters
    ----------
    image:
        Float input array.
    mask:
        Binary mask (same shape as *image*).
    strategy:
        ``"fixed_count"`` or ``"fixed_width"``.
    bin_count:
        Number of discrete grey levels (used when ``strategy="fixed_count"``).
    bin_width:
        Width of each bin (used when ``strategy="fixed_width"``).
    """
    disc = np.zeros_like(image, dtype=np.int32)
    roi_vals = image[mask > 0]

    if roi_vals.size == 0:
        return disc

    v_min = roi_vals.min()
    v_max = roi_vals.max()

    if strategy == "fixed_count":
        n_bins = bin_count
    elif strategy == "fixed_width":
        if bin_width is None:
            raise ValueError("bin_width must be provided when strategy='fixed_width'")
        n_bins = max(1, int(np.ceil((v_max - v_min) / bin_width)))
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy!r}")

    # Avoid division by zero when all values are identical
    if v_max == v_min:
        disc[mask > 0] = 1
        return disc

    # Map to [1, n_bins]
    indices = np.floor(
        n_bins * (image - v_min) / (v_max - v_min)
    ).astype(np.int32)
    # Clamp the maximum bin index
    indices = np.clip(indices, 1, n_bins)
    disc[mask > 0] = indices[mask > 0]
    return disc
