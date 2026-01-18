import numpy as np
import pytest

from src.data.preprocessing import (
    normalise_to_uint8,
    apply_morphological_transforms,
    get_largest_contour_bbox,
    get_breast_bbox,
    crop_to_breast,
)


class TestNormaliseToUint8:
    def test_already_uint8(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalise_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img)

    def test_float_image(self):
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalise_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[0, 127, 255]])

    def test_uint16_image(self):
        img = np.array([[0, 32768, 65535]], dtype=np.uint16)
        result = normalise_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[0, 127, 255]])

    def test_constant_image(self):
        img = np.full((10, 10), 100, dtype=np.float32)
        result = normalise_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, np.zeros((10, 10), dtype=np.uint8))

    def test_negative_values(self):
        img = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        result = normalise_to_uint8(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[0, 127, 255]])


class TestApplyMorphologicalTransforms:
    def test_removes_small_noise(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        mask[10:15, 10:15] = 255  # small noise

        result = apply_morphological_transforms(mask, kernel_size=20)

        # Main region should remain
        assert result[100, 100] == 255
        # Small noise should be removed
        assert result[12, 12] == 0

    def test_fills_small_holes(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:180, 20:180] = 255
        mask[95:105, 95:105] = 0  # small hole

        result = apply_morphological_transforms(mask, kernel_size=20)

        # Hole should be filled
        assert result[100, 100] == 255


class TestGetLargestContourBbox:
    def test_single_contour(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255

        x, y, w, h = get_largest_contour_bbox(mask)

        assert x == 30
        assert y == 20
        assert w == 40
        assert h == 60

    def test_multiple_contours_returns_largest(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # small
        mask[40:90, 40:90] = 255  # large

        x, y, w, h = get_largest_contour_bbox(mask)

        assert x == 40
        assert y == 40
        assert w == 50
        assert h == 50

    def test_empty_mask_returns_full_image(self):
        mask = np.zeros((100, 200), dtype=np.uint8)

        x, y, w, h = get_largest_contour_bbox(mask)

        assert x == 0
        assert y == 0
        assert w == 200
        assert h == 100


class TestGetBreastBbox:
    def test_bright_region_on_dark_background(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:150, 60:140] = 200

        x, y, w, h = get_breast_bbox(img, morph_kernel_size=10)

        assert 55 <= x <= 65
        assert 45 <= y <= 55
        assert 75 <= w <= 85
        assert 95 <= h <= 105

    def test_float_input(self):
        img = np.zeros((200, 200), dtype=np.float32)
        img[50:150, 60:140] = 1.0

        x, y, w, h = get_breast_bbox(img, morph_kernel_size=10)

        assert x >= 0
        assert y >= 0
        assert w > 0
        assert h > 0


class TestCropToBreast:
    def test_crops_to_bright_region(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:150, 60:140] = 200

        cropped = crop_to_breast(img, morph_kernel_size=10)

        assert cropped.shape[0] < 200
        assert cropped.shape[1] < 200
        assert cropped.mean() > img.mean()
