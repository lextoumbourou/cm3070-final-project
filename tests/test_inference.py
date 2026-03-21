"""
Tests for various inference functions.

Uses pytest style test.
"""

import numpy as np
import pytest

from src.inference_whole_image import (
    compute_metrics,
    get_inference_transform,
    get_tta_transforms,
)


class TestComputeMetrics:

    """Tests for compute_metrics function."""

    def test_single_class_labels_raises(self):
        """AUC is not defined when only one label."""
        probs = np.array([0.8, 0.6, 0.4])
        labels = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="AUC is undefined"):
            compute_metrics(probs, labels)

    def test_perfect_classifier(self):
        """Happy path test for metrics."""
        probs = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_metrics(probs, labels)
        assert metrics['auc'] == pytest.approx(1.0)
        assert metrics['sensitivity'] == pytest.approx(1.0)
        assert metrics['specificity'] == pytest.approx(1.0)
        assert metrics['accuracy'] == pytest.approx(1.0)


    def test_all_wrong_classifier(self):
        """Opposite happy path test for metrics."""
        probs = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_metrics(probs, labels)
        assert metrics['sensitivity'] == pytest.approx(0.0)
        assert metrics['specificity'] == pytest.approx(0.0)


    def test_all_positive_predictions(self):
        """All positive (perfect sensitivity)."""
        probs = np.array([0.9, 0.9, 0.9, 0.9])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_metrics(probs, labels)
        assert metrics['sensitivity'] == pytest.approx(1.0)
        assert metrics['specificity'] == pytest.approx(0.0)


    def test_all_negative_predictions(self):
        """All negative (perfect specificity)."""
        probs = np.array([0.1, 0.1, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_metrics(probs, labels)
        assert metrics['sensitivity'] == pytest.approx(0.0)
        assert metrics['specificity'] == pytest.approx(1.0)


class TestGetInferenceTransform:

    """Tests for get_inference_transform function."""

    def test_default_output_size(self):
        """Image is stretched to default."""
        transform = get_inference_transform()
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)['image']
        assert result.shape == (896, 1152, 3)

    def test_custom_output_size(self):
        """Image is resized based on input params."""
        transform = get_inference_transform(target_height=224, target_width=224)
        img = np.zeros((500, 400, 3), dtype=np.uint8)
        result = transform(image=img)['image']
        assert result.shape == (224, 224, 3)
