"""Types for the mammogram classifier app."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass
class FolderValidationResult:
    """Result of validating a training/test data folder."""

    error: str | None = None
    benign: int = 0
    malignant: int = 0
    total: int = 0
    benign_files: list[Path] = field(default_factory=list)
    malignant_files: list[Path] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Display information for a model."""

    name: str
    description: str
    is_vendor: bool = False
    is_default: bool = False


class Classification(Enum):
    """Binary classification result."""

    BENIGN = "Benign"
    MALIGNANT = "Malignant"


@dataclass
class InferenceResult:
    """Result of running inference on a single image."""

    malignant_prob: float
    classification: Classification


@dataclass
class EvaluationMetrics:
    """Metrics from evaluating a model on a dataset."""

    auc: float
    sensitivity: float
    specificity: float
    accuracy: float
    n_samples: int
    n_malignant: int
    n_benign: int


@dataclass
class TrainValSplit:
    """Result of splitting files into train/val sets."""

    train_benign: list[Path]
    train_malignant: list[Path]
    val_benign: list[Path]
    val_malignant: list[Path]
