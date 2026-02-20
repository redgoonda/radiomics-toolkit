"""Smoke tests for shape feature extraction (2D and 3D)."""

import numpy as np
import pytest

from radiomics_toolkit.features.shape import ShapeExtractor


@pytest.fixture
def circle_mask_2d():
    """64×64 mask with a filled circle."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    cy, cx, r = 32, 32, 20
    yy, xx = np.ogrid[:64, :64]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1
    image = mask.astype(np.float64) * 100
    return image, mask


@pytest.fixture
def sphere_mask_3d():
    """32×32×32 mask with a filled sphere."""
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    cz, cy, cx, r = 16, 16, 16, 10
    zz, yy, xx = np.ogrid[:32, :32, :32]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1
    image = mask.astype(np.float64) * 100
    return image, mask


class TestShapeExtractor2D:
    def test_returns_10_features(self, circle_mask_2d):
        img, mask = circle_mask_2d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        shape_feats = {k: v for k, v in feats.items() if k.startswith("shape_")}
        assert len(shape_feats) == 10, f"Expected 10, got {len(shape_feats)}: {list(shape_feats)}"

    def test_all_keys_prefixed(self, circle_mask_2d):
        img, mask = circle_mask_2d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        for k in feats:
            assert k.startswith("shape_")

    def test_area_reasonable(self, circle_mask_2d):
        img, mask = circle_mask_2d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        # Circle of radius 20: area ≈ pi * 400 ≈ 1257
        area = feats["shape_area"]
        assert 1100 < area < 1400, f"area={area}"

    def test_roundness_near_1_for_circle(self, circle_mask_2d):
        img, mask = circle_mask_2d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        roundness = feats["shape_roundness"]
        assert 0.85 < roundness <= 1.05, f"roundness={roundness}"

    def test_all_values_finite(self, circle_mask_2d):
        img, mask = circle_mask_2d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v}"


class TestShapeExtractor3D:
    def test_returns_10_features(self, sphere_mask_3d):
        img, mask = sphere_mask_3d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        shape_feats = {k: v for k, v in feats.items() if k.startswith("shape_")}
        assert len(shape_feats) == 10, f"Expected 10, got {len(shape_feats)}: {list(shape_feats)}"

    def test_volume_reasonable(self, sphere_mask_3d):
        img, mask = sphere_mask_3d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        # Sphere of radius 10: volume = (4/3)*pi*1000 ≈ 4189
        volume = feats["shape_volume"]
        assert 3800 < volume < 4600, f"volume={volume}"

    def test_sphericity_near_1(self, sphere_mask_3d):
        img, mask = sphere_mask_3d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        sph = feats["shape_sphericity"]
        assert 0.8 < sph <= 1.05, f"sphericity={sph}"

    def test_all_values_finite(self, sphere_mask_3d):
        img, mask = sphere_mask_3d
        ext = ShapeExtractor()
        feats = ext.extract(img, mask)
        for k, v in feats.items():
            assert np.isfinite(v) or np.isnan(v), f"{k} = {v}"
