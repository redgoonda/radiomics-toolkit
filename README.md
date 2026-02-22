# Radiomics Toolkit

A from-scratch Python library for quantitative imaging feature extraction, with a web UI, interactive DICOM viewer, and one-click auto-segmentation via TotalSegmentator.

## Features

- **83 radiomic features** across 6 classes: first-order statistics, shape, GLCM, GLRLM, GLSZM, NGTDM
- **DICOM viewer** — upload a DICOM series folder and browse axial slices with clinical windowing presets, interactive W/L adjustment, and scroll/keyboard navigation
- **Auto-segmentation** — click "Auto Segment" to generate an ROI mask with TotalSegmentator (117 anatomical structures for CT, 50 for MR) or Otsu thresholding for 2D images
- **Web UI** — drag-and-drop file or folder upload, mask overlay preview, interactive feature explorer, CSV/JSON export
- **CLI** — batch extraction to CSV or JSON
- **Python API** — use `RadiomicsExtractor` directly in your own scripts
- **Format support** — PNG, JPEG, TIFF, DICOM (single file or series folder), NIfTI (.nii / .nii.gz), NumPy .npy

## Install

```bash
git clone https://github.com/redgoonda/radiomics-toolkit.git
cd radiomics-toolkit
pip install -e ".[web]"

# Optional: 3D auto-segmentation support (downloads ~2 GB of model weights on first run)
pip install -e ".[segmentation]"
```

## Web UI

```bash
radiomics serve
# → http://127.0.0.1:8000
```

### DICOM Series Viewer

Drag a DICOM folder onto the image drop zone (or click **Upload DICOM Folder**). The top panel switches to a full-height viewer:

- **Slice navigation** — scrub the slider, click ◀▶, scroll the mouse wheel, or use arrow keys / Home / End
- **Windowing presets** — Abdomen (C:40 W:400), Lung (C:−600 W:1500), Bone (C:400 W:1800), Brain (C:40 W:80), Soft Tissue (C:50 W:350)
- **Interactive W/L** — click-drag on the image to adjust window centre (up/down) and window width (left/right), or type values directly into the C/W inputs
- **Corner overlays** — series name, current W/L, voxel spacing, and slice position

### Auto-Segmentation

Upload an image (any format), click **Auto Segment ⚡**, and a binary ROI mask is generated automatically. The mask is overlaid as a blue tint on the preview and stored in a server session — no re-upload needed when you click **Extract Features**.

## CLI

```bash
# Extract all features, output to CSV
radiomics extract --image scan.png --output features.csv

# With a mask, specific classes, custom bins
radiomics extract --image scan.nii.gz --mask mask.nii.gz \
  --features first_order,shape,glcm --bins 128 --output out.json

# Launch the web UI
radiomics serve --port 8000
```

## Python API

```python
from radiomics_toolkit import RadiomicsExtractor

extractor = RadiomicsExtractor(bin_count=64, normalize=False)

# From file (single image or DICOM directory)
results = extractor.extract_from_file("scan.png", mask_path="mask.png")
results = extractor.extract_from_file("dicom_series/")   # directory → 3D volume

# From numpy arrays
import numpy as np
image = np.load("image.npy")
mask  = np.load("mask.npy")
results = extractor.extract(image, mask)

print(f"{len(results)} features extracted")
# {'firstorder_mean': 142.3, 'shape_area': 576.0, 'glcm_contrast': 0.81, ...}
```

## Auto-Segmentation API

```python
from radiomics_toolkit.segmentation import segment_with_totalseg, segment_2d_fallback

# 3D — NIfTI file or DICOM directory
mask_3d = segment_with_totalseg("scan.nii.gz", fast=True, device="cpu")
mask_3d = segment_with_totalseg("dicom_series/")

# 2D — any numpy array (Otsu threshold + fill holes + remove small objects)
import numpy as np
mask_2d = segment_2d_fallback(np.array(...))
```

## Web API

The server exposes a REST API (docs at `/api/docs`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract` | POST | Extract features from image (+ optional mask or `session_id`) |
| `/segment` | POST | Auto-segment image, return `session_id` + preview PNG |
| `/preview` | POST | Return normalised PNG for display |
| `/dicom/session` | POST | Upload DICOM series, return viewer session metadata |
| `/dicom/{sid}/{index}` | GET | Fetch one windowed slice as PNG (`?wc=&ww=`) |

## Feature Classes

| Class | Features | Description |
|-------|----------|-------------|
| `first_order` | 18 | Mean, median, entropy, skewness, kurtosis, energy, … |
| `shape` | 8 | Area, perimeter, compactness, elongation, … |
| `glcm` | 24 | Contrast, correlation, homogeneity, ASM, … |
| `glrlm` | 11 | Run-length non-uniformity, grey-level variance, … |
| `glszm` | 13 | Zone entropy, size-zone non-uniformity, … |
| `ngtdm` | 5 | Coarseness, contrast, busyness, complexity, strength |

## Supported Formats

| Format | Image | Mask |
|--------|-------|------|
| PNG / JPEG / TIFF | ✓ | ✓ |
| DICOM (.dcm single file) | ✓ | ✓ |
| DICOM series (folder) | ✓ | ✓ |
| NIfTI (.nii, .nii.gz) | ✓ | ✓ |
| NumPy (.npy) | ✓ | ✓ |

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
