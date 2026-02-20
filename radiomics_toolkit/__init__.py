"""Radiomics Toolkit — quantitative imaging feature extraction.

Public API:
    RadiomicsExtractor  — main orchestrator
    load_image          — load image from file
    load_mask           — load mask from file
    write_results       — write feature dict to CSV or JSON
"""

from .extractor import RadiomicsExtractor
from .io.loaders import load_image, load_mask
from .io.writers import write_results

__version__ = "0.1.0"

__all__ = [
    "RadiomicsExtractor",
    "load_image",
    "load_mask",
    "write_results",
    "__version__",
]
