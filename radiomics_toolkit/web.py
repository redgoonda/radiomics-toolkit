"""FastAPI web server for the Radiomics Toolkit UI.

Start with:
    radiomics serve              (via CLI)
    python -m radiomics_toolkit.web   (direct)
"""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .extractor import RadiomicsExtractor, _ALL_CLASSES

app = FastAPI(title="Radiomics Toolkit", version="0.1.0", docs_url="/api/docs")

# Serve static files (index.html etc.)
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ---------------------------------------------------------------------------
# Server-side session store: session_id → Path to mask .npy file
# ---------------------------------------------------------------------------
_sessions: dict[str, Path] = {}

# DICOM viewer sessions: session_id → sorted list of .dcm Paths
_dicom_sessions: dict[str, list[Path]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_image_upload(
    image: Optional[UploadFile],
    dicom_files: List[UploadFile],
    tmp: Path,
) -> Path:
    """Save uploaded image(s) to *tmp* and return the path for loaders.

    - Single file  → ``tmp/<filename>``
    - Multiple DICOM files → ``tmp/dicom/`` directory (loader auto-detects)
    """
    if dicom_files:
        dcm_dir = tmp / "dicom"
        dcm_dir.mkdir(exist_ok=True)
        for f in dicom_files:
            fname = Path(f.filename).name if f.filename else f"slice_{uuid.uuid4().hex[:8]}.dcm"
            (dcm_dir / fname).write_bytes(await f.read())
        return dcm_dir
    else:
        fname = Path(image.filename).name if image.filename else "upload"
        p = tmp / fname
        p.write_bytes(await image.read())
        return p


@app.get("/", response_class=HTMLResponse)
async def root():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/preview")
async def preview(
    image: Optional[UploadFile] = File(None),
    dicom_files: List[UploadFile] = File(default=[]),
):
    """Return a normalised PNG for browser preview.

    Accepts either a single file (``image``) or multiple DICOM files
    (``dicom_files``).  For 3-D volumes the middle axial slice is returned.
    """
    from PIL import Image as PILImage

    with tempfile.TemporaryDirectory() as tmp:
        img_path = await _save_image_upload(image, dicom_files, Path(tmp))

        try:
            if img_path.is_dir():
                import pydicom

                dcm_files = sorted(img_path.glob("*.dcm"))
                if not dcm_files:
                    raise HTTPException(status_code=422, detail="No .dcm files in uploaded directory.")
                mid = dcm_files[len(dcm_files) // 2]
                ds = pydicom.dcmread(str(mid))
                arr = ds.pixel_array.astype(float)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept
            elif img_path.suffix.lower() == ".dcm":
                import pydicom

                ds = pydicom.dcmread(str(img_path))
                arr = ds.pixel_array.astype(float)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept
            else:
                arr = np.array(PILImage.open(str(img_path)).convert("L"), dtype=float)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    img_out = PILImage.fromarray(arr.astype(np.uint8))

    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/segment")
async def segment(
    image: Optional[UploadFile] = File(None),
    dicom_files: List[UploadFile] = File(default=[]),
    strategy: str = Form("totalsegmentator"),
    fast: bool = Form(True),
    device: str = Form("cpu"),
):
    """Auto-segment an image and return a session_id + preview overlay.

    Accepts either ``image`` (single file) or ``dicom_files`` (DICOM series).
    """
    from PIL import Image as PILImage

    if not image and not dicom_files:
        raise HTTPException(status_code=422, detail="Provide either 'image' or 'dicom_files'.")

    is_2d = False
    if image and not dicom_files:
        suffix = Path(image.filename or "upload").suffix.lower()
        is_2d = suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    session_dir = Path(tempfile.mkdtemp())
    sid = str(uuid.uuid4())

    try:
        tmp_in = Path(tempfile.mkdtemp())
        try:
            img_path = await _save_image_upload(image, dicom_files, tmp_in)

            if is_2d or strategy == "otsu":
                # ── 2D / Otsu path ─────────────────────────────────────
                from .segmentation import segment_2d_fallback

                try:
                    arr = np.array(
                        PILImage.open(str(img_path)).convert("L"), dtype=np.float64
                    )
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

                mask = segment_2d_fallback(arr)
                mask_path = session_dir / "mask.npy"
                np.save(str(mask_path), mask)
                _sessions[sid] = mask_path

                preview_b64 = _make_2d_overlay_b64(arr, mask)
                return JSONResponse(content={
                    "session_id": sid,
                    "voxels": int(mask.sum()),
                    "dims": list(mask.shape),
                    "method": "otsu",
                    "preview": preview_b64,
                })

            else:
                # ── 3D / TotalSegmentator path ──────────────────────────
                from .segmentation import segment_with_totalseg

                try:
                    mask = segment_with_totalseg(img_path, fast=fast, device=device)
                except Exception as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"TotalSegmentator failed: {exc}",
                    )

                mask_path = session_dir / "mask.npy"
                np.save(str(mask_path), mask)
                _sessions[sid] = mask_path

                # Middle axial slice preview
                if mask.ndim == 3:
                    mid = mask.shape[0] // 2
                    mask_slice = mask[mid]
                    try:
                        import nibabel as nib
                        img_nib = nib.load(str(img_path))
                        img_arr = np.asarray(img_nib.dataobj)
                        img_slice = img_arr[mid].astype(float) if img_arr.ndim == 3 else img_arr.astype(float)
                    except Exception:
                        img_slice = mask_slice.astype(float) * 128
                    preview_b64 = _make_2d_overlay_b64(img_slice, mask_slice)
                else:
                    preview_b64 = _make_2d_overlay_b64(mask.astype(float) * 128, mask)

                return JSONResponse(content={
                    "session_id": sid,
                    "voxels": int(mask.sum()),
                    "dims": list(mask.shape),
                    "method": "totalsegmentator",
                    "preview": preview_b64,
                })
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _make_2d_overlay_b64(image: np.ndarray, mask: np.ndarray) -> str:
    """Render a greyscale image with a blue mask overlay; return base64 PNG."""
    from PIL import Image as PILImage

    arr = image.astype(float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    else:
        arr = np.zeros_like(arr)

    h, w = arr.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    grey = arr.astype(np.uint8)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = 255

    m = mask > 0
    rgba[m, 0] = np.clip(rgba[m, 0].astype(int) * 60 // 100 + 91  * 40 // 100, 0, 255).astype(np.uint8)
    rgba[m, 1] = np.clip(rgba[m, 1].astype(int) * 60 // 100 + 141 * 40 // 100, 0, 255).astype(np.uint8)
    rgba[m, 2] = np.clip(rgba[m, 2].astype(int) * 60 // 100 + 238 * 40 // 100, 0, 255).astype(np.uint8)

    img_out = PILImage.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/extract")
async def extract(
    image: Optional[UploadFile] = File(None),
    dicom_files: List[UploadFile] = File(default=[]),
    mask: Optional[UploadFile] = File(None),
    bins: int = Form(64),
    normalize: bool = Form(False),
    feature_classes: str = Form(",".join(_ALL_CLASSES)),
    session_id: str = Form(""),
):
    """Extract features from an image (single file or DICOM directory)."""
    if not image and not dicom_files:
        raise HTTPException(status_code=422, detail="Provide either 'image' or 'dicom_files'.")

    classes = [c.strip() for c in feature_classes.split(",") if c.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_path = await _save_image_upload(image, dicom_files, tmp)

        mask_path = None
        if mask and mask.filename:
            mask_path = tmp / mask.filename
            mask_path.write_bytes(await mask.read())
        elif session_id and session_id in _sessions:
            stored = _sessions[session_id]
            if stored.exists():
                mask_path = tmp / "session_mask.npy"
                shutil.copy2(stored, mask_path)

        try:
            extractor = RadiomicsExtractor(
                bin_count=bins,
                feature_classes=classes,
                normalize=normalize,
            )
            results = extractor.extract_from_file(img_path, mask_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    clean: dict[str, float | None] = {}
    for k, v in results.items():
        try:
            fv = float(v)
            clean[k] = None if not np.isfinite(fv) else fv
        except (TypeError, ValueError):
            clean[k] = None

    return JSONResponse(content=clean)


# ---------------------------------------------------------------------------
# DICOM viewer endpoints
# ---------------------------------------------------------------------------

@app.post("/dicom/session")
async def create_dicom_session(dicom_files: List[UploadFile] = File(...)):
    """Upload a DICOM series and create a viewer session.

    Returns metadata (slice count, matrix size, pixel spacing, default W/L)
    and a ``session_id`` used to fetch individual slices.
    """
    import pydicom

    if not dicom_files:
        raise HTTPException(status_code=422, detail="No DICOM files provided.")

    session_dir = Path(tempfile.mkdtemp())
    sid = str(uuid.uuid4())

    # Save all files
    saved: list[Path] = []
    for f in dicom_files:
        fname = Path(f.filename).name if f.filename else f"{uuid.uuid4().hex}.dcm"
        p = session_dir / fname
        p.write_bytes(await f.read())
        saved.append(p)

    # Sort by InstanceNumber
    def _instance_num(path: Path) -> int:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    saved.sort(key=_instance_num)
    _dicom_sessions[sid] = saved

    # Read metadata from the first slice
    rows, cols = 512, 512
    pixel_spacing = [1.0, 1.0]
    slice_thickness = 1.0
    wc, ww = 40.0, 400.0
    study_desc = series_desc = ""

    try:
        ds = pydicom.dcmread(str(saved[0]), stop_before_pixels=True)
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        pixel_spacing = [float(ps[0]), float(ps[1])]
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
        study_desc  = str(getattr(ds, "StudyDescription", ""))
        series_desc = str(getattr(ds, "SeriesDescription", ""))

        raw_wc = getattr(ds, "WindowCenter", 40)
        raw_ww = getattr(ds, "WindowWidth", 400)
        # Tags can be multi-value sequences
        wc = float(raw_wc[0] if hasattr(raw_wc, "__iter__") and not isinstance(raw_wc, str) else raw_wc)
        ww = float(raw_ww[0] if hasattr(raw_ww, "__iter__") and not isinstance(raw_ww, str) else raw_ww)
    except Exception:
        pass

    return JSONResponse(content={
        "session_id":        sid,
        "slice_count":       len(saved),
        "rows":              rows,
        "cols":              cols,
        "pixel_spacing":     pixel_spacing,
        "slice_thickness":   slice_thickness,
        "default_wc":        wc,
        "default_ww":        ww,
        "study_description": study_desc,
        "series_description": series_desc,
    })


@app.get("/dicom/{session_id}/{index}")
async def get_dicom_slice(
    session_id: str,
    index: int,
    wc: float = 40.0,
    ww: float = 400.0,
    mask_session: str = "",
):
    """Return a single DICOM slice as a windowed PNG.

    Query params:
      ``wc``           Window centre (HU).
      ``ww``           Window width (HU).
      ``mask_session`` Optional seg session_id — overlays the mask in blue.
    """
    import pydicom
    from PIL import Image as PILImage

    if session_id not in _dicom_sessions:
        raise HTTPException(status_code=404, detail="DICOM session not found.")

    slices = _dicom_sessions[session_id]
    if index < 0 or index >= len(slices):
        raise HTTPException(status_code=404, detail=f"Slice index {index} out of range (0–{len(slices)-1}).")

    try:
        ds  = pydicom.dcmread(str(slices[index]))
        arr = ds.pixel_array.astype(np.float64)
        slope     = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read slice: {exc}")

    # Apply window / level → 0–255
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0

    # Build RGBA canvas
    grey = arr.astype(np.uint8)
    rgba = np.stack([grey, grey, grey, np.full_like(grey, 255)], axis=-1)

    # Overlay segmentation mask if requested
    if mask_session and mask_session in _sessions:
        mask_path = _sessions[mask_session]
        if mask_path.exists():
            full_mask = np.load(str(mask_path))
            mask_slice: np.ndarray | None = None
            if full_mask.ndim == 3 and index < full_mask.shape[0]:
                mask_slice = full_mask[index]
            elif full_mask.ndim == 2:
                mask_slice = full_mask
            if mask_slice is not None:
                m = mask_slice > 0
                rgba[m, 0] = np.clip(rgba[m, 0].astype(int) * 60 // 100 + 91  * 40 // 100, 0, 255).astype(np.uint8)
                rgba[m, 1] = np.clip(rgba[m, 1].astype(int) * 60 // 100 + 141 * 40 // 100, 0, 255).astype(np.uint8)
                rgba[m, 2] = np.clip(rgba[m, 2].astype(int) * 60 // 100 + 238 * 40 // 100, 0, 255).astype(np.uint8)

    img = PILImage.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

def main(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to run the web server: pip install uvicorn"
        ) from exc
    uvicorn.run(
        "radiomics_toolkit.web:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
