"""Shared type definitions for training modules."""

from dataclasses import dataclass


@dataclass
class BinaryValidationMetrics:
    """Validation metrics for binary classification tasks."""
    val_loss: float
    val_accuracy: float
    val_auc: float
    probs: list[float] | None = None
    targets: list[int] | None = None
