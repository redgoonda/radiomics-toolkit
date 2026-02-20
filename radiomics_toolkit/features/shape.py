"""Shape features — 2D morphology and 3D morphology via marching cubes."""

from __future__ import annotations

import numpy as np

from .base import FeatureExtractor


class ShapeExtractor(FeatureExtractor):
    """Compute shape/morphological features.

    Automatically selects 2D or 3D features based on ``image.ndim``.
    """

    prefix = "shape"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        if mask.ndim == 2:
            return self._extract_2d(mask)
        elif mask.ndim == 3:
            return self._extract_3d(mask)
        else:
            raise ValueError(f"Unsupported mask dimensionality: {mask.ndim}")

    # ------------------------------------------------------------------
    # 2D features
    # ------------------------------------------------------------------

    def _extract_2d(self, mask: np.ndarray) -> dict[str, float]:
        try:
            from skimage.measure import regionprops, label
        except ImportError as exc:
            raise ImportError("scikit-image is required for 2D shape features") from exc

        labeled = label(mask)
        props_list = regionprops(labeled)

        if not props_list:
            nan = float("nan")
            return self._prefixed({
                "area": nan, "perimeter": nan, "equivalent_diameter": nan,
                "major_axis_length": nan, "minor_axis_length": nan,
                "elongation": nan, "eccentricity": nan, "roundness": nan,
                "extent": nan, "convex_hull_ratio": nan,
            })

        # Use largest region
        props = max(props_list, key=lambda p: p.area)

        area = float(props.area)
        perimeter = float(props.perimeter) if props.perimeter > 0 else 1e-9
        equiv_diam = float(props.equivalent_diameter)
        major = float(props.major_axis_length) if props.major_axis_length > 0 else 1e-9
        minor = float(props.minor_axis_length)
        eccentricity = float(props.eccentricity)

        elongation = minor / major if major > 0 else float("nan")
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else float("nan")
        extent = float(props.extent)
        convex_hull_ratio = area / float(props.convex_area) if props.convex_area > 0 else float("nan")

        features = {
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": equiv_diam,
            "major_axis_length": major,
            "minor_axis_length": minor,
            "elongation": elongation,
            "eccentricity": eccentricity,
            "roundness": roundness,
            "extent": extent,
            "convex_hull_ratio": convex_hull_ratio,
        }
        return self._prefixed(features)

    # ------------------------------------------------------------------
    # 3D features
    # ------------------------------------------------------------------

    def _extract_3d(self, mask: np.ndarray) -> dict[str, float]:
        try:
            from skimage.measure import marching_cubes, mesh_surface_area
        except ImportError as exc:
            raise ImportError("scikit-image is required for 3D shape features") from exc

        nan = float("nan")

        # Volume (voxel count with unit spacing)
        volume = float(np.sum(mask > 0))

        if volume == 0:
            return self._prefixed({
                "volume": 0.0, "surface_area": nan, "sphericity": nan,
                "compactness": nan, "surface_to_volume_ratio": nan,
                "max_3d_diameter": nan, "major_axis_length": nan,
                "minor_axis_length": nan, "least_axis_length": nan,
                "flatness": nan,
            })

        # Surface area via marching cubes
        try:
            verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5)
            surface_area = float(mesh_surface_area(verts, faces))
        except Exception:
            surface_area = nan

        # Sphericity = ratio of sphere surface area to actual surface area
        # sphere with same volume has SA = (pi^(1/3)) * (6*V)^(2/3)
        if surface_area and surface_area > 0:
            sphere_sa = (np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3)
            sphericity = float(sphere_sa / surface_area)
            compactness = float((36 * np.pi * volume ** 2) / (surface_area ** 3))
            sv_ratio = float(surface_area / volume)
        else:
            sphericity = nan
            compactness = nan
            sv_ratio = nan

        # Max 3D diameter: max Euclidean distance between any two ROI voxels
        # Approximate using convex hull vertices for efficiency
        coords = np.argwhere(mask > 0).astype(float)
        max_diam = _max_diameter(coords)

        # Axis lengths from inertia tensor eigenvectors
        major, minor, least = _axis_lengths_3d(coords)

        flatness = least / major if major > 0 else nan

        features = {
            "volume": volume,
            "surface_area": surface_area,
            "sphericity": sphericity,
            "compactness": compactness,
            "surface_to_volume_ratio": sv_ratio,
            "max_3d_diameter": max_diam,
            "major_axis_length": major,
            "minor_axis_length": minor,
            "least_axis_length": least,
            "flatness": flatness,
        }
        return self._prefixed(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_diameter(coords: np.ndarray) -> float:
    """Approximate maximum pairwise Euclidean distance via convex hull."""
    if coords.shape[0] < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]
    except Exception:
        hull_pts = coords

    # Brute-force on (possibly reduced) hull points
    max_d2 = 0.0
    n = hull_pts.shape[0]
    for i in range(n):
        diff = hull_pts[i + 1:] - hull_pts[i]
        d2 = np.sum(diff ** 2, axis=1)
        if d2.size > 0:
            max_d2 = max(max_d2, float(d2.max()))
    return float(np.sqrt(max_d2))


def _axis_lengths_3d(coords: np.ndarray) -> tuple[float, float, float]:
    """Return (major, minor, least) axis lengths from the inertia tensor."""
    if coords.shape[0] < 2:
        return 0.0, 0.0, 0.0

    centroid = coords.mean(axis=0)
    c = coords - centroid
    # Covariance matrix / inertia tensor
    cov = (c.T @ c) / (c.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    # Axis lengths ~ 4 * sqrt(eigenvalue) (2-sigma span)
    lengths = 4 * np.sqrt(eigvals)
    lengths_sorted = np.sort(lengths)[::-1]  # descending
    return float(lengths_sorted[0]), float(lengths_sorted[1]), float(lengths_sorted[2])
