"""Auto-segmentation helpers for Radiomics Toolkit.

Provides:
  - segment_with_totalseg: 3D segmentation via TotalSegmentator
  - segment_2d_fallback:   Simple Otsu-based 2D segmentation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def segment_with_totalseg(
    image_path: str | Path,
    fast: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Run TotalSegmentator on a NIfTI or DICOM directory and return a binary mask.

    Parameters
    ----------
    image_path:
        Path to a NIfTI file (.nii / .nii.gz) or a DICOM directory.
    fast:
        Use fast (3 mm) mode when True; standard (1.5 mm) when False.
    device:
        Compute device: "cpu" or "gpu" (or a torch device string).

    Returns
    -------
    np.ndarray
        Binary uint8 array with shape matching the input volume.
        All segmented structures are merged: (seg_data > 0).astype(np.uint8).
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "TotalSegmentator is required for 3D segmentation: "
            "pip install TotalSegmentator"
        ) from exc

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for loading NIfTI files: pip install nibabel"
        ) from exc

    image_path = Path(image_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "segmentation"
        output_path.mkdir()

        totalsegmentator(
            image_path,
            output_path,
            ml=True,
            fast=fast,
            device=device,
            skip_saving=False,
        )

        # TotalSegmentator writes a multilabel NIfTI when ml=True
        nifti_files = list(output_path.glob("*.nii.gz")) + list(output_path.glob("*.nii"))
        if not nifti_files:
            raise RuntimeError(
                f"TotalSegmentator produced no NIfTI output in {output_path}"
            )

        seg_img = nib.load(str(nifti_files[0]))
        seg_data = np.asarray(seg_img.dataobj)

    return (seg_data > 0).astype(np.uint8)


def segment_2d_fallback(image: np.ndarray) -> np.ndarray:
    """Simple Otsu-based segmentation for 2D images.

    Used when TotalSegmentator is unavailable or the input is a 2D image
    (TotalSegmentator only handles 3D volumes).

    Parameters
    ----------
    image:
        2D numpy array (grayscale pixel values).

    Returns
    -------
    np.ndarray
        Binary uint8 mask (same shape as *image*).
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy.ndimage import binary_fill_holes

    arr = image.astype(np.float64)

    # Otsu threshold
    thresh = threshold_otsu(arr)
    binary = arr > thresh

    # Fill holes
    filled = binary_fill_holes(binary)

    # Remove small spurious objects (min_size = 1% of total pixels, at least 64)
    min_size = max(64, int(filled.size * 0.01))
    cleaned = remove_small_objects(filled, min_size=min_size)

    return cleaned.astype(np.uint8)
