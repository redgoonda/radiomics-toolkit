"""Abstract base class for all feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    """Base class for radiomics feature extractors.

    Subclasses must implement :meth:`extract` and set :attr:`prefix`.
    """

    #: Short prefix prepended to every feature name, e.g. ``"firstorder"``.
    prefix: str = ""

    @abstractmethod
    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Compute features from *image* within *mask*.

        Parameters
        ----------
        image:
            Original (or discretized) numpy array.
        mask:
            Binary mask (same shape as *image*); non-zero voxels are in the ROI.

        Returns
        -------
        dict
            Mapping ``feature_name`` → scalar float value.
            Names should NOT include the prefix (it will be added by the
            orchestrator).
        """

    def _prefixed(self, features: dict[str, float]) -> dict[str, float]:
        """Return *features* with keys prefixed by ``self.prefix + '_'``."""
        if not self.prefix:
            return features
        return {f"{self.prefix}_{k}": v for k, v in features.items()}
