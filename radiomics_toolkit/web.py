"""FastAPI web server for the Radiomics Toolkit UI.

Start with:
    radiomics serve              (via CLI)
    python -m radiomics_toolkit.web   (direct)
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

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


@app.get("/", response_class=HTMLResponse)
async def root():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    strategy: str = Form("totalsegmentator"),
    fast: bool = Form(True),
    device: str = Form("cpu"),
):
    """Auto-segment an image and return a session_id + preview overlay.

    Form fields
    -----------
    strategy : "totalsegmentator" | "otsu"
    fast     : bool — use fast (3 mm) TotalSegmentator mode
    device   : str  — compute device for TotalSegmentator ("cpu" / "gpu")
    image    : uploaded image file
    """
    from PIL import Image as PILImage

    data = await image.read()
    suffix = Path(image.filename or "upload").suffix.lower()
    is_2d = suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    # ------------------------------------------------------------------
    # Create a temp dir that persists beyond this request (session-owned)
    # ------------------------------------------------------------------
    session_dir = Path(tempfile.mkdtemp())
    sid = str(uuid.uuid4())

    try:
        if is_2d or strategy == "otsu":
            # ── 2D / Otsu path ─────────────────────────────────────────
            from .segmentation import segment_2d_fallback

            try:
                arr = np.array(
                    PILImage.open(io.BytesIO(data)).convert("L"), dtype=np.float64
                )
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

            mask = segment_2d_fallback(arr)

            # Save mask to session
            mask_path = session_dir / "mask.npy"
            np.save(str(mask_path), mask)
            _sessions[sid] = mask_path

            # Build overlay preview
            preview_b64 = _make_2d_overlay_b64(arr, mask)

            return JSONResponse(
                content={
                    "session_id": sid,
                    "voxels": int(mask.sum()),
                    "dims": list(mask.shape),
                    "method": "otsu",
                    "preview": preview_b64,
                }
            )

        else:
            # ── 3D / TotalSegmentator path ──────────────────────────────
            from .segmentation import segment_with_totalseg

            # Write the uploaded file to disk so TotalSegmentator can read it
            img_path = session_dir / (image.filename or f"image{suffix}")
            img_path.write_bytes(data)

            try:
                mask = segment_with_totalseg(img_path, fast=fast, device=device)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"TotalSegmentator failed: {exc}",
                )

            # Save mask to session
            mask_path = session_dir / "mask.npy"
            np.save(str(mask_path), mask)
            _sessions[sid] = mask_path

            # Extract middle axial slice for preview
            if mask.ndim == 3:
                mid = mask.shape[0] // 2
                mask_slice = mask[mid]
                # Load the original image middle slice for overlay
                try:
                    import nibabel as nib

                    img_nib = nib.load(str(img_path))
                    img_arr = np.asarray(img_nib.dataobj)
                    if img_arr.ndim == 3:
                        img_slice = img_arr[mid].astype(float)
                    else:
                        img_slice = img_arr.astype(float)
                except Exception:
                    img_slice = mask_slice.astype(float) * 128
                preview_b64 = _make_2d_overlay_b64(img_slice, mask_slice)
            else:
                preview_b64 = _make_2d_overlay_b64(
                    mask.astype(float) * 128, mask
                )

            return JSONResponse(
                content={
                    "session_id": sid,
                    "voxels": int(mask.sum()),
                    "dims": list(mask.shape),
                    "method": "totalsegmentator",
                    "preview": preview_b64,
                }
            )

    except HTTPException:
        # Clean up session dir on error
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _make_2d_overlay_b64(image: np.ndarray, mask: np.ndarray) -> str:
    """Render a greyscale image with a blue mask overlay; return base64 PNG."""
    from PIL import Image as PILImage

    # Normalise image to 0–255
    arr = image.astype(float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    else:
        arr = np.zeros_like(arr)

    # RGBA canvas
    h, w = arr.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    grey = arr.astype(np.uint8)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = 255

    # Stamp mask in blue tint rgba(91,141,238,0.4)
    m = mask > 0
    rgba[m, 0] = np.clip(rgba[m, 0].astype(int) * 60 // 100 + 91 * 40 // 100, 0, 255).astype(np.uint8)
    rgba[m, 1] = np.clip(rgba[m, 1].astype(int) * 60 // 100 + 141 * 40 // 100, 0, 255).astype(np.uint8)
    rgba[m, 2] = np.clip(rgba[m, 2].astype(int) * 60 // 100 + 238 * 40 // 100, 0, 255).astype(np.uint8)

    img_out = PILImage.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    bins: int = Form(64),
    normalize: bool = Form(False),
    feature_classes: str = Form(",".join(_ALL_CLASSES)),
    session_id: str = Form(""),
):
    """Upload image (+ optional mask or session_id), return extracted features as JSON."""
    classes = [c.strip() for c in feature_classes.split(",") if c.strip()]

    # Save uploads to temp files so loaders can handle them normally
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / image.filename
        img_path.write_bytes(await image.read())

        mask_path = None

        # Priority 1: explicit mask upload
        if mask and mask.filename:
            mask_path = Path(tmp) / mask.filename
            mask_path.write_bytes(await mask.read())

        # Priority 2: session mask from /segment
        elif session_id and session_id in _sessions:
            stored = _sessions[session_id]
            if stored.exists():
                # Copy into the temp dir so the loader can find it
                mask_path = Path(tmp) / "session_mask.npy"
                import shutil as _shutil
                _shutil.copy2(stored, mask_path)

        try:
            extractor = RadiomicsExtractor(
                bin_count=bins,
                feature_classes=classes,
                normalize=normalize,
            )
            results = extractor.extract_from_file(img_path, mask_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # Convert numpy scalars → Python floats for JSON serialisation
    clean: dict[str, float | None] = {}
    for k, v in results.items():
        try:
            fv = float(v)
            clean[k] = None if not np.isfinite(fv) else fv
        except (TypeError, ValueError):
            clean[k] = None

    return JSONResponse(content=clean)


@app.post("/preview")
async def preview(image: UploadFile = File(...)):
    """Return image as a normalised PNG for browser preview."""
    from PIL import Image as PILImage

    data = await image.read()
    suffix = Path(image.filename).suffix.lower()

    try:
        if suffix == ".dcm":
            import pydicom

            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
                f.write(data)
                f.flush()
                ds = pydicom.dcmread(f.name)
            arr = ds.pixel_array.astype(float)
        else:
            arr = np.array(PILImage.open(io.BytesIO(data)).convert("L"), dtype=float)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    # Normalise to 0–255 for display
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    img_out = PILImage.fromarray(arr.astype(np.uint8))

    buf = io.BytesIO()
    img_out.save(buf, format="PNG")
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
