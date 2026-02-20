"""Smoke tests for first-order feature extraction."""

import numpy as np
import pytest

from radiomics_toolkit.features.first_order import FirstOrderExtractor


@pytest.fixture
def simple_2d_image():
    """Small 10×10 image with known statistics."""
    rng = np.random.default_rng(42)
    img = rng.integers(10, 200, size=(10, 10)).astype(np.float64)
    mask = np.ones((10, 10), dtype=np.uint8)
    return img, mask


@pytest.fixture
def circle_image():
    """64×64 image with a circular ROI."""
    img = np.zeros((64, 64), dtype=np.float64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cy, cx, r = 32, 32, 20
    yy, xx = np.ogrid[:64, :64]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    rng = np.random.default_rng(0)
    img[circle] = rng.uniform(50, 200, size=circle.sum())
    mask[circle] = 1
    return img, mask


class TestFirstOrderExtractor:
    def test_returns_all_23_features(self, simple_2d_image):
        img, mask = simple_2d_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        assert len(feats) == 23, f"Expected 23 features, got {len(feats)}: {list(feats.keys())}"

    def test_all_keys_prefixed(self, simple_2d_image):
        img, mask = simple_2d_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        for k in feats:
            assert k.startswith("firstorder_"), f"Key {k!r} missing 'firstorder_' prefix"

    def test_all_values_finite(self, simple_2d_image):
        img, mask = simple_2d_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k!r} is not finite: {v}"

    def test_mean_correct(self, simple_2d_image):
        img, mask = simple_2d_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        expected_mean = float(np.mean(img[mask > 0]))
        assert abs(feats["firstorder_mean"] - expected_mean) < 1e-9

    def test_energy_correct(self, simple_2d_image):
        img, mask = simple_2d_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        expected_energy = float(np.sum(img[mask > 0] ** 2))
        assert abs(feats["firstorder_energy"] - expected_energy) < 1e-6

    def test_circle_roi(self, circle_image):
        img, mask = circle_image
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        assert len(feats) == 23
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v}"

    def test_empty_mask_returns_nan(self):
        img = np.zeros((10, 10), dtype=np.float64)
        mask = np.zeros((10, 10), dtype=np.uint8)
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        for v in feats.values():
            assert np.isnan(v)

    def test_3d_image(self):
        rng = np.random.default_rng(7)
        img = rng.uniform(0, 100, (20, 20, 20))
        mask = np.ones((20, 20, 20), dtype=np.uint8)
        ext = FirstOrderExtractor()
        feats = ext.extract(img, mask)
        assert len(feats) == 23
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
