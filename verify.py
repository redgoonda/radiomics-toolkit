"""End-to-end verification script for the radiomics toolkit.

Run after `pip install -e .`:
    python verify.py
"""

import sys
import numpy as np

# ── 1. Imports ───────────────────────────────────────────────────────────────
print("1. Importing radiomics_toolkit …", end=" ")
from radiomics_toolkit import RadiomicsExtractor, __version__
print(f"OK  (v{__version__})")

# ── 2. 2D phantom ────────────────────────────────────────────────────────────
print("2. Building 2D circle phantom …", end=" ")
img2d = np.zeros((128, 128), dtype=np.float64)
mask2d = np.zeros((128, 128), dtype=np.uint8)
yy, xx = np.ogrid[:128, :128]
circle = (yy - 64) ** 2 + (xx - 64) ** 2 <= 40 ** 2
rng = np.random.default_rng(42)
img2d[circle] = rng.uniform(50, 200, size=circle.sum())
mask2d[circle] = 1
print("OK")

print("3. Extracting 2D features …", end=" ")
ext = RadiomicsExtractor(bin_count=32)
results2d = ext.extract(img2d, mask2d)
print(f"OK  ({len(results2d)} features)")

# Check all values are finite floats
bad = {k: v for k, v in results2d.items() if not (isinstance(v, float) and np.isfinite(v))}
if bad:
    print(f"   WARNING — non-finite values: {bad}")
else:
    print("   All 2D feature values are finite floats ✓")

# ── 3. 3D phantom ────────────────────────────────────────────────────────────
print("4. Building 3D sphere phantom …", end=" ")
img3d = np.zeros((40, 40, 40), dtype=np.float64)
mask3d = np.zeros((40, 40, 40), dtype=np.uint8)
zz, yy, xx = np.ogrid[:40, :40, :40]
sphere = (zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= 12 ** 2
img3d[sphere] = rng.uniform(50, 200, size=sphere.sum())
mask3d[sphere] = 1
print("OK")

print("5. Extracting 3D features …", end=" ")
results3d = ext.extract(img3d, mask3d)
print(f"OK  ({len(results3d)} features)")

bad3d = {k: v for k, v in results3d.items() if not (isinstance(v, float) and np.isfinite(v))}
if bad3d:
    print(f"   WARNING — non-finite values: {bad3d}")
else:
    print("   All 3D feature values are finite floats ✓")

# ── 4. Save phantom PNG and run CLI end-to-end ───────────────────────────────
print("6. Saving phantom PNG …", end=" ")
from pathlib import Path
from PIL import Image as PILImage

fixtures = Path("tests/fixtures")
fixtures.mkdir(parents=True, exist_ok=True)
phantom_path = fixtures / "phantom_2d.png"
PILImage.fromarray(img2d.astype(np.uint8)).save(phantom_path)
print(f"OK  ({phantom_path})")

print("7. Running CLI (radiomics extract) …")
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "radiomics_toolkit.cli", "extract",
     "--image", str(phantom_path.resolve()),
     "--output", str(Path("out.csv").resolve()),
     "--bins", "32"],
    capture_output=True, text=True,
    cwd=str(Path(__file__).parent),
)
if result.returncode != 0:
    print(f"   CLI FAILED (exit {result.returncode}):")
    print(f"   stdout: {result.stdout.strip()}")
    print(f"   stderr: {result.stderr.strip()}")
else:
    print(f"   {result.stdout.strip()}")
    # Validate CSV
    import pandas as pd
    df = pd.read_csv("out.csv")
    print(f"   CSV has {df.shape[1]} columns, {df.shape[0]} row")
    bad_csv = [c for c in df.columns if not np.isfinite(df[c].iloc[0])]
    if bad_csv:
        print(f"   WARNING — non-finite columns: {bad_csv[:10]}")
    else:
        print("   All CSV values are finite ✓")

# ── 5. Feature class inventory ───────────────────────────────────────────────
print("\n── Feature summary ──────────────────────────────────")
prefixes = {}
for k in results2d:
    prefix = k.split("_")[0]
    prefixes[prefix] = prefixes.get(prefix, 0) + 1
for prefix, count in sorted(prefixes.items()):
    print(f"  {prefix:15s} {count:3d} features")
print(f"  {'TOTAL':15s} {len(results2d):3d} features")

print("\nVerification complete.")
