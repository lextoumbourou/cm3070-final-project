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


def test_single_class_labels_raises(self):
    """AUC is not defined when only one label."""
    probs = np.array([0.8, 0.6, 0.4])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="AUC is undefined"):
        compute_metrics(probs, labels)

