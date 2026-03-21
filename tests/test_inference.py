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
