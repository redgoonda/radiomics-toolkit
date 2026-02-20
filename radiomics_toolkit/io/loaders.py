"""Image and mask loaders.

Supported formats:
- Single DICOM file (.dcm)
- Directory of DICOM files → 3D volume (sorted by InstanceNumber)
- PNG / JPEG / TIFF / BMP via Pillow → 2D array
- NumPy .npy files
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from *path* and return a float64 numpy array.

    Dispatch rules (in order):
    1. If *path* is a directory → load as DICOM series (3-D).
    2. If *path* ends with .dcm → load single DICOM slice.
    3. If *path* ends with .npy → load numpy array directly.
    4. Otherwise → Pillow (PNG / JPEG / TIFF / BMP) → 2-D greyscale.
    """
    path = Path(path)

    if path.is_dir():
        return _load_dicom_series(path)

    suffix = path.suffix.lower()

    if suffix == ".dcm":
        return _load_single_dicom(path)

    if suffix == ".npy":
        arr = np.load(str(path))
        return arr.astype(np.float64)

    return _load_pillow(path)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a binary mask from *path* and return a uint8 numpy array."""
    path = Path(path)
    suffix = path.suffix.lower()

    if path.is_dir():
        arr = _load_dicom_series(path)
    elif suffix == ".dcm":
        arr = _load_single_dicom(path)
    elif suffix == ".npy":
        arr = np.load(str(path))
    else:
        arr = _load_pillow(path)

    # Binarise: anything > 0 becomes 1
    mask = (arr > 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_single_dicom(path: Path) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to load DICOM files: pip install pydicom") from exc

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float64)

    # Apply rescale slope/intercept when present (Hounsfield units, etc.)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return arr


def _load_dicom_series(directory: Path) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to load DICOM series: pip install pydicom") from exc

    dcm_files = sorted(directory.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in directory: {directory}")

    # Read all slices and sort by InstanceNumber
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        slices.append(ds)

    slices.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))

    # Stack into 3-D volume
    volume = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume.append(arr * slope + intercept)

    return np.stack(volume, axis=0)


def _load_pillow(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for PNG/JPEG/TIFF loading: pip install Pillow") from exc

    img = Image.open(str(path))
    # Convert to greyscale (luminance)
    img = img.convert("L")
    return np.array(img, dtype=np.float64)
