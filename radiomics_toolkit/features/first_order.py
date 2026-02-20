"""First-order statistical features (23 features, 2D/3D agnostic)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from .base import FeatureExtractor
from ..utils import get_voxels_in_mask


class FirstOrderExtractor(FeatureExtractor):
    """Compute 23 first-order statistics from the ROI intensity histogram."""

    prefix = "firstorder"

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        voxels = get_voxels_in_mask(image, mask)

        if voxels.size == 0:
            return {k: float("nan") for k in self._feature_names()}

        n = voxels.size
        mean = float(np.mean(voxels))
        std = float(np.std(voxels, ddof=1)) if n > 1 else 0.0
        variance = float(np.var(voxels, ddof=1)) if n > 1 else 0.0
        median = float(np.median(voxels))

        # Mode (most frequent bin using 64-bin histogram)
        hist, bin_edges = np.histogram(voxels, bins=64)
        mode_bin = int(np.argmax(hist))
        mode = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)

        v_min = float(np.min(voxels))
        v_max = float(np.max(voxels))
        v_range = v_max - v_min

        # Skewness and kurtosis (excess, Fisher definition)
        skewness = float(scipy_skew(voxels)) if n > 2 else 0.0
        kurt = float(scipy_kurtosis(voxels, fisher=True)) if n > 3 else 0.0

        # Energy = sum of squared voxel values
        energy = float(np.sum(voxels ** 2))

        # Total energy (energy scaled by voxel volume; here voxel spacing = 1 mm³)
        total_energy = energy  # spacing assumed 1

        # Entropy (from 64-bin normalised histogram)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

        # Root mean square
        rms = float(np.sqrt(np.mean(voxels ** 2)))

        # Mean absolute deviation
        mad = float(np.mean(np.abs(voxels - mean)))

        # Robust MAD (10th–90th percentile values only)
        p10, p90 = np.percentile(voxels, [10, 90])
        robust_vals = voxels[(voxels >= p10) & (voxels <= p90)]
        robust_mad = float(np.mean(np.abs(robust_vals - np.mean(robust_vals)))) if robust_vals.size > 0 else 0.0

        # Inter-quartile range
        p25, p75 = np.percentile(voxels, [25, 75])
        iqr = float(p75 - p25)

        # Uniformity (sum of squared histogram probabilities)
        hist2, _ = np.histogram(voxels, bins=64)
        probs2 = hist2 / hist2.sum()
        uniformity = float(np.sum(probs2 ** 2))

        # Coefficient of variation
        cov = float(std / mean) if mean != 0 else float("nan")

        features = {
            "mean": mean,
            "median": median,
            "mode": mode,
            "minimum": v_min,
            "maximum": v_max,
            "range": v_range,
            "variance": variance,
            "std_dev": std,
            "skewness": skewness,
            "kurtosis": kurt,
            "energy": energy,
            "entropy": entropy,
            "rms": rms,
            "mad": mad,
            "robust_mad": robust_mad,
            "iqr": iqr,
            "10th_percentile": float(p10),
            "90th_percentile": float(p90),
            "total_energy": total_energy,
            "uniformity": uniformity,
            "coefficient_of_variation": cov,
            "interquartile_range": iqr,
            "mean_absolute_deviation": mad,
        }
        return self._prefixed(features)

    def _feature_names(self) -> list[str]:
        return [
            f"{self.prefix}_{n}"
            for n in [
                "mean", "median", "mode", "minimum", "maximum", "range",
                "variance", "std_dev", "skewness", "kurtosis", "energy",
                "entropy", "rms", "mad", "robust_mad", "iqr",
                "10th_percentile", "90th_percentile", "total_energy",
                "uniformity", "coefficient_of_variation",
                "interquartile_range", "mean_absolute_deviation",
            ]
        ]
