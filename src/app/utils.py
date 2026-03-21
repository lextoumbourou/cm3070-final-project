"""Utility functions for the mammogram classifier app."""

import random
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pydicom
from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.app.types import (
    Classification,
    EvaluationMetrics,
    FolderValidationResult,
    InferenceResult,
    ModelInfo,
    TrainValSplit,
)
from src.models.whole_image_classifier import create_whole_image_classifier

MODEL_DESCRIPTIONS: dict[str, ModelInfo] = {
    "cbis-whole-wd-only": ModelInfo(
        name="CBIS-DDSM Base Model (Default)",
        description="Trained on CBIS-DDSM dataset. Use this as a starting point for fine-tuning.",
        is_vendor=True,
        is_default=True,
    ),
    "cbis-whole-final": ModelInfo(
        name="CBIS-DDSM Final",
        description=(
            "Trained on CBIS-DDSM train+val set. "
            "Slightly higher AUC but used test-set selection."
        ),
        is_vendor=True,
    ),
    "inbreast-whole-finetune": ModelInfo(
        name="INbreast Fine-tuned",
        description="Fine-tuned on INbreast dataset (Portuguese FFDM).",
        is_vendor=True,
    ),
    "vindr-balanced-finetune": ModelInfo(
        name="VinDr Fine-tuned",
        description="Fine-tuned on VinDr-Mammo dataset (Vietnamese hospitals).",
        is_vendor=True,
    ),
}


def get_model_display_info(weights_path: str) -> ModelInfo:
    """Get display name and description for a model."""
    model_name = Path(weights_path).parent.name
    if model_name in MODEL_DESCRIPTIONS:
        return MODEL_DESCRIPTIONS[model_name]
    return ModelInfo(name=model_name, description="User fine-tuned model")


def load_model(weights_path: str | None):
    """Load a whole image classifier model."""
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name="resnet50",
        num_classes=2,
    )
    if weights_path and Path(weights_path).exists():
        model.load_weights(weights_path)
    model.eval()
    return model


def load_dicom(file_bytes: bytes) -> np.ndarray:
    """Load DICOM file bytes and return as RGB numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    dcm = pydicom.dcmread(tmp_path)
    pixel_array = dcm.pixel_array
    if pixel_array.dtype != np.uint8:
        pixel_array = (
            (pixel_array - pixel_array.min())
            / (pixel_array.max() - pixel_array.min())
            * 255
        ).astype(np.uint8)
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    Path(tmp_path).unlink()
    return pixel_array


def run_inference(model, img: mx.array) -> InferenceResult:
    """Run inference on a single preprocessed image."""
    inputs = mx.expand_dims(img, 0)
    logits = model(inputs)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    malignant_prob = probs[0, 1].item()
    classification = Classification.MALIGNANT if malignant_prob >= 0.5 else Classification.BENIGN
    return InferenceResult(malignant_prob=malignant_prob, classification=classification)


def get_confidence_description(malignant_prob: float) -> tuple[str, str]:
    """Get a textual description of model confidence."""
    confidence = abs(malignant_prob - 0.5) * 2

    if confidence >= 0.8:
        level = "Very high"
        explanation = "The model is very certain about this prediction."
    elif confidence >= 0.5:
        level = "High"
        explanation = "The model is fairly certain about this prediction."
    elif confidence >= 0.2:
        level = "Moderate"
        explanation = "The model shows reasonable certainty."
    else:
        level = "Low"
        explanation = "The model has low confidence. Consider additional review."

    return level, explanation


def validate_training_folder(folder_path: str) -> FolderValidationResult:
    """Validate a training/test folder and return stats or error."""
    folder = Path(folder_path)
    if not folder.exists():
        return FolderValidationResult(error="Folder does not exist")

    benign_folder = folder / "benign"
    malignant_folder = folder / "malignant"

    if not benign_folder.exists() or not malignant_folder.exists():
        return FolderValidationResult(
            error="Folder must contain 'benign/' and 'malignant/' subfolders"
        )

    image_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    benign_images = [
        f for f in benign_folder.iterdir() if f.suffix.lower() in image_extensions
    ]
    malignant_images = [
        f for f in malignant_folder.iterdir() if f.suffix.lower() in image_extensions
    ]

    if len(benign_images) == 0 or len(malignant_images) == 0:
        return FolderValidationResult(error="Both folders must contain at least one image")

    return FolderValidationResult(
        benign=len(benign_images),
        malignant=len(malignant_images),
        total=len(benign_images) + len(malignant_images),
        benign_files=benign_images,
        malignant_files=malignant_images,
    )


def stratified_train_val_split(
    benign_files: list[Path],
    malignant_files: list[Path],
    val_fraction: float = 0.2,
) -> TrainValSplit:
    """Split files into train/val sets, maintaining class balance."""
    random.shuffle(benign_files)
    random.shuffle(malignant_files)

    n_benign_val = max(1, int(len(benign_files) * val_fraction))
    n_malignant_val = max(1, int(len(malignant_files) * val_fraction))

    return TrainValSplit(
        train_benign=benign_files[n_benign_val:],
        train_malignant=malignant_files[n_malignant_val:],
        val_benign=benign_files[:n_benign_val],
        val_malignant=malignant_files[:n_malignant_val],
    )


def evaluate_model(model, dataset) -> EvaluationMetrics:
    """Evaluate model on a dataset, returning AUC, sensitivity, specificity."""
    model.eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []

    for inputs, targets in loader:
        logits = model(inputs)
        probs = mx.softmax(logits, axis=1)
        mx.eval(probs)

        all_probs.extend(probs[:, 1].tolist())
        all_labels.extend(targets.tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = 0.5 if len(np.unique(all_labels)) < 2 else roc_auc_score(all_labels, all_probs)

    preds = (all_probs >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (all_labels == 1))
    tn = np.sum((preds == 0) & (all_labels == 0))
    fp = np.sum((preds == 1) & (all_labels == 0))
    fn = np.sum((preds == 0) & (all_labels == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0

    return EvaluationMetrics(
        auc=float(auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        accuracy=float(accuracy),
        n_samples=len(all_labels),
        n_malignant=int(np.sum(all_labels)),
        n_benign=int(np.sum(all_labels == 0)),
    )


class FolderDataset(Dataset):
    """Dataset for loading images from benign/malignant file lists."""

    def __init__(self, benign_files, malignant_files, transform=None):
        self.transform = transform
        self.samples = []
        for f in benign_files:
            self.samples.append((str(f), 0))
        for f in malignant_files:
            self.samples.append((str(f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        if filepath.lower().endswith((".dcm", ".dicom")):
            dcm = pydicom.dcmread(filepath)
            img = dcm.pixel_array
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
        else:
            img = Image.open(filepath).convert("RGB")
            img = np.array(img).astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32) / 255.0
        return mx.array(img), mx.array(label)


def run_finetuning(model, dataset, epochs: int, stage1_epochs: int, progress_callback):
    """Run fine-tuning on a model with two-stage training."""
    model.freeze_backbone()
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.001)

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    total_batches = len(loader) * epochs
    current_batch = 0

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        return mx.mean(nn.losses.cross_entropy(logits, targets))

    for epoch in range(epochs):
        if epoch == stage1_epochs:
            model.unfreeze_all()
            optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for inputs, targets in loader:
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            logits = model(inputs)
            preds = mx.argmax(logits, axis=1)
            correct = mx.sum(preds == targets).item()

            batch_size = len(targets)
            epoch_loss += loss.item() * batch_size
            epoch_correct += correct
            epoch_samples += batch_size
            current_batch += 1

            progress = current_batch / total_batches
            stage = "Stage 1 (head only)" if epoch < stage1_epochs else "Stage 2 (full model)"
            acc = epoch_correct / epoch_samples if epoch_samples > 0 else 0
            progress_callback(progress, epoch + 1, epochs, stage, epoch_loss / epoch_samples, acc)

    model.eval()
    return model
