# Radiomics Toolkit

A from-scratch Python library for quantitative imaging feature extraction, with a web UI and one-click auto-segmentation via TotalSegmentator.

## Features

- **83 radiomic features** across 6 classes: first-order statistics, shape, GLCM, GLRLM, GLSZM, NGTDM
- **Auto-segmentation** — click "Auto Segment" to generate an ROI mask with TotalSegmentator (117 anatomical structures for CT, 50 for MR) or Otsu thresholding for 2D images
- **Web UI** — drag-and-drop image upload, live preview with mask overlay, interactive feature explorer, CSV/JSON export
- **CLI** — batch extraction to CSV or JSON
- **Python API** — use `RadiomicsExtractor` directly in your own scripts
- **Format support** — PNG, JPEG, TIFF, DICOM, NIfTI (.nii / .nii.gz), NumPy .npy

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

Upload an image, click **Auto Segment ⚡** to generate a mask, then **Extract Features**.

The overlay renders the mask as a blue tint directly on the image preview. For 3D volumes (NIfTI/DICOM), the middle axial slice is shown.

## CLI

```bash
# Extract all features, output to CSV
radiomics extract --image scan.png --output features.csv

# With a mask, specific classes, custom bins
radiomics extract --image scan.nii.gz --mask mask.nii.gz \
  --features first_order,shape,glcm --bins 128 --output out.json
```

## Python API

```python
from radiomics_toolkit import RadiomicsExtractor

extractor = RadiomicsExtractor(bin_count=64, normalize=False)

# From file
results = extractor.extract_from_file("scan.png", mask_path="mask.png")

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

# 3D — NIfTI or DICOM directory
mask_3d = segment_with_totalseg("scan.nii.gz", fast=True, device="cpu")

# 2D — any numpy array
import numpy as np
image = np.array(...)
mask_2d = segment_2d_fallback(image)
```

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
| DICOM (.dcm) | ✓ | ✓ |
| NIfTI (.nii, .nii.gz) | ✓ | ✓ |
| NumPy (.npy) | ✓ | ✓ |

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
