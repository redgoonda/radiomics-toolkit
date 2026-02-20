"""Shared helpers: direction vectors, padding, neighbourhood utilities."""

import numpy as np


def get_directions_2d() -> list[tuple[int, int]]:
    """Return the 4 unique 2D directions (0°, 45°, 90°, 135°)."""
    return [(0, 1), (1, 1), (1, 0), (1, -1)]


def get_directions_3d() -> list[tuple[int, int, int]]:
    """Return the 13 unique 3D directions (half-sphere of offsets)."""
    dirs = []
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                # Keep only half-sphere to avoid duplicate pairs
                if (dz > 0) or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                    dirs.append((dz, dy, dx))
    return dirs  # 13 directions


def pad_image(image: np.ndarray, pad_width: int = 1) -> np.ndarray:
    """Pad image with zeros (for neighbourhood computations)."""
    return np.pad(image, pad_width, mode="constant", constant_values=0)


def get_voxels_in_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return 1D array of image values where mask > 0."""
    return image[mask > 0].astype(float)


def check_mask_shape(image: np.ndarray, mask: np.ndarray) -> None:
    """Raise if image and mask shapes differ."""
    if image.shape != mask.shape:
        raise ValueError(
            f"Image shape {image.shape} does not match mask shape {mask.shape}"
        )


def make_full_mask(image: np.ndarray) -> np.ndarray:
    """Return an all-ones uint8 mask covering the full image."""
    return np.ones(image.shape, dtype=np.uint8)
