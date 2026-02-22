# CLAUDE.md — Radiomics Toolkit

Project-level instructions and context for Claude Code sessions.

## Project Overview

**radiomics-toolkit** — from-scratch Python radiomics feature extraction library.
Repo: https://github.com/redgoonda/radiomics-toolkit

## Key File Locations

```
radiomics_toolkit/
  extractor.py        # RadiomicsExtractor — main orchestrator
  segmentation.py     # segment_with_totalseg, segment_2d_fallback
  web.py              # FastAPI server, /extract, /segment, /preview endpoints
  cli.py              # Click CLI (radiomics extract / radiomics serve)
  io/
    loaders.py        # load_image, load_mask (PNG/JPEG/TIFF/DICOM/NIfTI/npy)
    writers.py        # write_results (CSV / JSON)
  preprocessing/
    discretizer.py    # fixed-count grey-level discretization
  features/
    first_order.py    # 18 features
    shape.py          # 8 features
    glcm.py           # 24 features
    glrlm.py          # 11 features
    glszm.py          # 13 features
    ngtdm.py          # 5 features  → 83 total
  static/
    index.html        # Single-page web UI
tests/
  test_first_order.py
  test_shape.py
pyproject.toml        # Build config + optional deps
```

## Architecture

- **RadiomicsExtractor** (`extractor.py`) — accepts image + mask numpy arrays or file paths, runs discretization, dispatches to feature extractors, returns flat dict
- **Feature extractors** (`features/`) — each implements `.extract(image, mask) → dict`
- **Web server** (`web.py`) — FastAPI; `_sessions` dict maps `session_id → Path` to persisted `.npy` mask; `/segment` runs segmentation and returns `session_id + base64 preview`; `/extract` accepts either an uploaded mask or a `session_id`
- **Segmentation** (`segmentation.py`) — `segment_with_totalseg` for 3D NIfTI/DICOM, `segment_2d_fallback` (Otsu + fill holes + remove small objects) for 2D

## Install

```bash
pip install -e ".[web]"           # core + web UI
pip install -e ".[segmentation]"  # adds TotalSegmentator + nibabel
pip install -e ".[dev]"           # adds pytest
```

## Running

```bash
radiomics serve                   # web UI at http://127.0.0.1:8000
radiomics extract -i img.png -o out.csv
python3 -m radiomics_toolkit.web  # alternative server start
```

## Testing

```bash
pytest tests/

# Quick smoke tests
python3 -c "
from radiomics_toolkit.segmentation import segment_2d_fallback
import numpy as np
img = np.zeros((64,64)); img[20:44,20:44] = 150
m = segment_2d_fallback(img)
assert m.shape == (64,64) and m.dtype == np.uint8 and m.max() == 1
print('OK', m.sum(), 'voxels')
"
```

## Optional Dependencies

```toml
[project.optional-dependencies]
web          = ["fastapi", "uvicorn", "python-multipart"]
segmentation = ["TotalSegmentator", "nibabel>=3.2"]
dev          = ["pytest>=7.0"]
```

## `gh` CLI

Installed at `~/bin/gh` (v2.87.2). Already authenticated as `redgoonda`.
Add `~/bin` to PATH if needed: `export PATH="$HOME/bin:$PATH"`

## Conventions

- Feature extractors return keys prefixed with the class name: `firstorder_mean`, `shape_area`, `glcm_contrast`, etc.
- Masks are always binary uint8; image arrays are float64 inside extractors
- `_sessions` in `web.py` is in-memory only — restarting the server clears all sessions
- TotalSegmentator model weights (~2 GB) are downloaded on first use to `~/.totalsegmentator/`
