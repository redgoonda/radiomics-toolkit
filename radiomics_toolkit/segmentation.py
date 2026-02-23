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
    # --- pydicom 2.x / dicom2nifti compatibility shim -----------------------
    # dicom2nifti (a TotalSegmentator dependency) does
    #   from pydicom.pixels import apply_modality_lut
    # at module-import time.  pydicom.pixels only exists in pydicom ≥3.0,
    # which requires Python ≥3.10.  On Python 3.9 we only have pydicom 2.x,
    # so we inject a lightweight fake module before TotalSegmentator is first
    # imported.
    import sys, types as _types
    if "pydicom.pixels" not in sys.modules:
        import pydicom as _pydicom
        from pydicom.pixel_data_handlers.util import (
            apply_modality_lut as _apply_modality_lut,
        )
        _px = _types.ModuleType("pydicom.pixels")
        _px.apply_modality_lut = _apply_modality_lut
        sys.modules["pydicom.pixels"] = _px
        _pydicom.pixels = _px          # also patch the attribute on the package
    # -------------------------------------------------------------------------

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
        tmp_dir = Path(tmp_dir)

        # When ml=True, TotalSegmentator expects output to be a *file path*
        # (it calls nib.save(result, output) directly).  Pass a .nii.gz path.
        output_nii = tmp_dir / "segmentation.nii.gz"

        # TotalSegmentator's DICOM ingestion uses pydicom.pixels (3.x only).
        # On Python <3.10 we only have pydicom 2.x, so convert DICOM → NIfTI
        # ourselves first, then hand TotalSegmentator a plain .nii.gz file.
        if image_path.is_dir():
            image_path = _dicom_dir_to_nifti(image_path, tmp_dir / "input.nii.gz")

        totalsegmentator(
            image_path,
            output_nii,
            ml=True,
            fast=fast,
            device=device,
            skip_saving=False,
        )

        if not output_nii.exists():
            raise RuntimeError(
                f"TotalSegmentator produced no output at {output_nii}"
            )

        seg_img = nib.load(str(output_nii))
        seg_data = np.asarray(seg_img.dataobj)

    return (seg_data > 0).astype(np.uint8)


def _dicom_dir_to_nifti(dicom_dir: Path, out_path: Path) -> Path:
    """Convert a DICOM series directory to a NIfTI file using pydicom 2.x + nibabel.

    TotalSegmentator uses ``pydicom.pixels`` internally (pydicom 3.x only).
    This helper lets us hand it a plain .nii.gz on Python ≤3.9 where only
    pydicom 2.x is available.

    Parameters
    ----------
    dicom_dir:
        Directory containing .dcm files for a single series.
    out_path:
        Destination .nii.gz path (parent must already exist).

    Returns
    -------
    Path
        ``out_path`` after the file has been written.
    """
    import pydicom
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for DICOM→NIfTI conversion: pip install nibabel"
        ) from exc

    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        # Also try without extension (some DICOM stores have no .dcm suffix)
        dcm_files = [
            p for p in dicom_dir.iterdir()
            if p.is_file() and not p.suffix.lower() in {".json", ".txt", ".csv"}
        ]
    if not dcm_files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")

    # Read all slices, sort by InstanceNumber (or ImagePositionPatient z)
    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise RuntimeError(f"Could not read any DICOM files from {dicom_dir}")

    def _sort_key(ds):
        if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
            return float(ds.InstanceNumber)
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return 0.0

    slices.sort(key=_sort_key)

    # Stack pixel data into a 3D volume applying rescale slope/intercept
    frames = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        frames.append(arr * slope + intercept)

    volume = np.stack(frames, axis=0)  # shape: (Z, Y, X)

    # Build affine from DICOM geometry tags
    ds0 = slices[0]
    pixel_spacing = [1.0, 1.0]
    if hasattr(ds0, "PixelSpacing"):
        pixel_spacing = [float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])]
    slice_thickness = 1.0
    if hasattr(ds0, "SliceThickness") and ds0.SliceThickness:
        slice_thickness = float(ds0.SliceThickness)
    elif len(slices) > 1:
        # Compute from z-positions
        def _z(ds):
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            return 0.0
        dz = abs(_z(slices[1]) - _z(slices[0]))
        if dz > 0:
            slice_thickness = dz

    # NIfTI affine: diagonal with voxel sizes; axes mapped to RAS
    affine = np.diag([
        pixel_spacing[1],   # X (col) → R
        pixel_spacing[0],   # Y (row) → A
        slice_thickness,    # Z (slice) → S
        1.0,
    ])

    # nibabel NIfTI convention: (X, Y, Z) — transpose from (Z, Y, X)
    volume_nib = volume.transpose(2, 1, 0)

    img = nib.Nifti1Image(volume_nib, affine)
    nib.save(img, str(out_path))
    return out_path


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
