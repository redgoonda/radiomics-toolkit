"""FastAPI web server for the Radiomics Toolkit UI.

Start with:
    radiomics serve              (via CLI)
    python -m radiomics_toolkit.web   (direct)
"""

from __future__ import annotations

import io
import json
import tempfile
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


@app.get("/", response_class=HTMLResponse)
async def root():
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    bins: int = Form(64),
    normalize: bool = Form(False),
    feature_classes: str = Form(",".join(_ALL_CLASSES)),
):
    """Upload image (+ optional mask), return extracted features as JSON."""
    classes = [c.strip() for c in feature_classes.split(",") if c.strip()]

    # Save uploads to temp files so loaders can handle them normally
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / image.filename
        img_path.write_bytes(await image.read())

        mask_path = None
        if mask and mask.filename:
            mask_path = Path(tmp) / mask.filename
            mask_path.write_bytes(await mask.read())

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
