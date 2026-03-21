"""Basic tests for shared transforms module."""

import numpy as np

from src.transforms import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    get_inference_transform,
    get_train_transform,
    get_tta_transforms,
    preprocess_image,
)


class TestGetInferenceTransform:
    def test_default_dimensions(self):
        """Uses default dimensions when no args provided."""
        transform = get_inference_transform()
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

    def test_custom_dimensions(self):
        """Resizes to custom dimensions."""
        transform = get_inference_transform(height=224, width=224)
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (224, 224, 3)


class TestGetTrainTransform:
    def test_output_size(self):
        """Output matches requested dimensions."""
        transform = get_train_transform(height=224, width=224)
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (224, 224, 3)

    def test_default_dimensions(self):
        """Uses default dimensions when no args provided."""
        transform = get_train_transform()
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)["image"]
        assert result.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)


class TestGetTtaTransforms:
    def test_returns_four_transforms(self):
        """Returns exactly 4 TTA variants."""
        transforms = get_tta_transforms()
        assert len(transforms) == 4

    def test_all_variants_produce_correct_size(self):
        """Each TTA variant resizes to target dimensions."""
        transforms = get_tta_transforms(height=224, width=224)
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        for t in transforms:
            result = t(image=img)["image"]
            assert result.shape == (224, 224, 3)


class TestPreprocessImage:
    def test_output_shape(self):
        """Output shape matches transform dimensions."""
        transform = get_inference_transform(height=224, width=224)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_image(img, transform)
        assert result.shape == (224, 224, 3)

    def test_normalization_range(self):
        """Output values are normalized to [0, 1]."""
        transform = get_inference_transform(height=100, width=100)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(img, transform)
        result_np = np.array(result)
        assert result_np.min() >= 0.0
        assert result_np.max() <= 1.0
