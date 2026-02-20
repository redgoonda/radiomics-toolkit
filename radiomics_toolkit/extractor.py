"""RadiomicsExtractor — orchestrates loading, preprocessing, and feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .io.loaders import load_image, load_mask
from .preprocessing.discretizer import discretize
from .utils import make_full_mask, check_mask_shape
from .features.first_order import FirstOrderExtractor
from .features.shape import ShapeExtractor
from .features.glcm import GLCMExtractor
from .features.glrlm import GLRLMExtractor
from .features.glszm import GLSZMExtractor
from .features.ngtdm import NGTDMExtractor


_ALL_CLASSES = ["first_order", "shape", "glcm", "glrlm", "glszm", "ngtdm"]

_EXTRACTOR_MAP = {
    "first_order": FirstOrderExtractor,
    "shape": ShapeExtractor,
    "glcm": GLCMExtractor,
    "glrlm": GLRLMExtractor,
    "glszm": GLSZMExtractor,
    "ngtdm": NGTDMExtractor,
}


class RadiomicsExtractor:
    """High-level orchestrator for radiomics feature extraction.

    Parameters
    ----------
    bin_count:
        Number of grey levels for discretization (default 64).
    feature_classes:
        List of feature class names to compute. ``None`` means all classes.
        Valid values: ``"first_order"``, ``"shape"``, ``"glcm"``,
        ``"glrlm"``, ``"glszm"``, ``"ngtdm"``.
    normalize:
        If ``True``, normalize the image to [0, 1] before processing.
    """

    def __init__(
        self,
        bin_count: int = 64,
        feature_classes: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> None:
        self.bin_count = bin_count
        self.normalize = normalize

        if feature_classes is None:
            self.feature_classes = list(_ALL_CLASSES)
        else:
            unknown = set(feature_classes) - set(_ALL_CLASSES)
            if unknown:
                raise ValueError(f"Unknown feature classes: {unknown}")
            self.feature_classes = list(feature_classes)

        # Instantiate extractors
        self._extractors = {
            name: _EXTRACTOR_MAP[name]()
            for name in self.feature_classes
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Extract features from *image* within *mask*.

        Parameters
        ----------
        image:
            Input image as a numpy array (2D or 3D).
        mask:
            Binary mask (same shape as *image*). If ``None``, the entire
            image is used as the ROI.

        Returns
        -------
        dict
            Flat dictionary mapping ``prefix_feature_name`` → float.
        """
        image = image.astype(np.float64)

        if mask is None:
            mask = make_full_mask(image)
        else:
            mask = mask.astype(np.uint8)

        check_mask_shape(image, mask)

        if self.normalize:
            roi_vals = image[mask > 0]
            v_min, v_max = roi_vals.min(), roi_vals.max()
            if v_max > v_min:
                image = (image - v_min) / (v_max - v_min)

        # Discretize for texture features
        disc_image = discretize(image, mask, strategy="fixed_count", bin_count=self.bin_count)

        results: dict[str, float] = {}

        for name, extractor in self._extractors.items():
            # Shape uses the original image (for geometry), others use discretized
            if name == "first_order":
                feats = extractor.extract(image, mask)
            elif name == "shape":
                feats = extractor.extract(image, mask)
            else:
                feats = extractor.extract(disc_image, mask)
            results.update(feats)

        return results

    def extract_from_file(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Load files and extract features.

        Parameters
        ----------
        image_path:
            Path to image file or DICOM directory.
        mask_path:
            Path to mask file. If ``None``, the entire image is used.
        """
        image = load_image(image_path)
        mask = load_mask(mask_path) if mask_path is not None else None
        return self.extract(image, mask)

    def to_dataframe(self, results: dict[str, float]):
        """Convert *results* dict to a single-row pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required: pip install pandas") from exc
        return pd.DataFrame([results])
