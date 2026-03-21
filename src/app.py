"""Streamlit web interface for mammogram classification and fine-tuning."""

import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from sklearn.metrics import roc_auc_score

# Ensure mlx-image is in path.
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlx-image" / "src"))
from mlxim.data import DataLoader
from mlxim.data._base import Dataset

from src.models.whole_image_classifier import create_whole_image_classifier
from src.transforms import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    get_inference_transform,
    get_train_transform,
    preprocess_image,
)

DEFAULT_WEIGHTS = "checkpoints/default/cbis-whole-wd-only/best_model.safetensors"

# Emoji prefixes for model selection UI
VENDOR_MODEL_PREFIX = "📦 "
USER_MODEL_PREFIX = "👤 "

TRAINING_PRESETS = {
    "Quick": {"epochs": 5, "stage1_epochs": 2},
    "Standard": {"epochs": 15, "stage1_epochs": 5},
    "Thorough": {"epochs": 30, "stage1_epochs": 10},
}

# Model descriptions for UI display
MODEL_DESCRIPTIONS = {
    "cbis-whole-wd-only": {
        "name": "CBIS-DDSM Base Model (Default)",
        "description": (
            "Trained on CBIS-DDSM dataset. "
            "Use this as a starting point for fine-tuning."),
        "is_default": True,
        "vendor": True,
    },
    "cbis-whole-final": {
        "name": "CBIS-DDSM Final",
        "description": (
            "Trained on CBIS-DDSM train+val set. "
            "Slightly higher AUC but used test-set selection."),
        "is_default": False,
        "vendor": True,
    },
    "inbreast-whole-finetune": {
        "name": "INbreast Fine-tuned",
        "description": "Fine-tuned on INbreast dataset (Portuguese FFDM).",
        "is_default": False,
        "vendor": True,
    },
    "vindr-whole-finetune": {
        "name": "VinDr Fine-tuned",
        "description": "Fine-tuned on VinDr-Mammo dataset (Vietnamese hospitals).",
        "is_default": False,
        "vendor": True,
    },
}


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
    is_vendor: bool


class Classification(Enum):
    """Binary classification result."""

    BENIGN = "Benign"
    MALIGNANT = "Malignant"


@dataclass
class InferenceResult:
    """Result of running inference on a single image."""

    malignant_prob: float
    classification: Classification


def get_model_display_info(weights_path: str) -> ModelInfo:
    """Get display name and description for a model."""
    model_name = Path(weights_path).parent.name
    if model_name in MODEL_DESCRIPTIONS:
        info = MODEL_DESCRIPTIONS[model_name]
        return ModelInfo(
            name=info["name"],
            description=info["description"],
            is_vendor=info.get("vendor", False),
        )
    else:
        # User-trained model
        return ModelInfo(name=model_name, description="User fine-tuned model", is_vendor=False)


def load_model(weights_path):
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name="resnet50",
        num_classes=2
    )
    if weights_path and Path(weights_path).exists():
        model.load_weights(weights_path)
    model.eval()
    return model


def load_dicom(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    dcm = pydicom.dcmread(tmp_path)
    pixel_array = dcm.pixel_array
    if pixel_array.dtype != np.uint8:
        pixel_array = ((pixel_array - pixel_array.min()) /
                       (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)
    Path(tmp_path).unlink()
    return pixel_array


def run_inference(model, img) -> InferenceResult:
    """Run inference on a single preprocessed image."""
    inputs = mx.expand_dims(img, 0)
    logits = model(inputs)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    malignant_prob = probs[0, 1].item()
    classification = Classification.MALIGNANT if malignant_prob >= 0.5 else Classification.BENIGN
    return InferenceResult(malignant_prob=malignant_prob, classification=classification)


def get_confidence_description(malignant_prob):
    """Get a textual description of model confidence."""
    # Confidence is distance from decision boundary, scaled to 0-1
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

    image_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
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


def stratified_train_val_split(benign_files, malignant_files, val_fraction=0.2):
    """Split files into train/val sets, maintaining class balance."""
    np.random.shuffle(benign_files)
    np.random.shuffle(malignant_files)

    n_benign_val = max(1, int(len(benign_files) * val_fraction))
    n_malignant_val = max(1, int(len(malignant_files) * val_fraction))

    val_benign = benign_files[:n_benign_val]
    train_benign = benign_files[n_benign_val:]

    val_malignant = malignant_files[:n_malignant_val]
    train_malignant = malignant_files[n_malignant_val:]

    return train_benign, train_malignant, val_benign, val_malignant


def evaluate_model(model, dataset):
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

    # AUC (0.5 fallback when single class present)
    auc = 0.5 if len(np.unique(all_labels)) < 2 else roc_auc_score(all_labels, all_probs)

    # Sensitivity and specificity at threshold 0.5
    preds = (all_probs >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (all_labels == 1))
    tn = np.sum((preds == 0) & (all_labels == 0))
    fp = np.sum((preds == 1) & (all_labels == 0))
    fn = np.sum((preds == 0) & (all_labels == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "n_samples": len(all_labels),
        "n_malignant": int(np.sum(all_labels)),
        "n_benign": int(np.sum(all_labels == 0)),
    }


class FolderDataset(Dataset):
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

        if filepath.lower().endswith(('.dcm', '.dicom')):
            dcm = pydicom.dcmread(filepath)
            img = dcm.pixel_array
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
        else:
            img = Image.open(filepath).convert('RGB')
            img = np.array(img).astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32) / 255.0
        return mx.array(img), mx.array(label)


def run_finetuning(model, dataset, epochs, stage1_epochs, progress_callback):
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


def inference_tab():
    st.header("Inference")

    weights_path = st.session_state.get("current_weights", DEFAULT_WEIGHTS)

    if "model" not in st.session_state or st.session_state.get("model_weights") != weights_path:
        if Path(weights_path).exists():
            with st.spinner("Loading model..."):
                st.session_state.model = load_model(weights_path)
                st.session_state.model_weights = weights_path
        else:
            st.error(f"Weights not found: {weights_path}")
            return

    model = st.session_state.model
    transform = get_inference_transform()
    model_info = get_model_display_info(weights_path)

    st.caption(f"Using model: **{model_info.name}**")

    # Batch Inference on Test Dataset.
    test_folder = st.session_state.get("test_folder", "")
    test_stats = st.session_state.get("test_stats", None)

    if test_folder and test_stats:
        st.subheader("Batch Inference on Test Dataset")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"Test folder: `{test_folder}` ({test_stats.total} images)")
        with col2:
            run_batch = st.button("Run Evaluation", type="primary", use_container_width=True)

        if run_batch:
            with st.spinner("Running inference on test dataset..."):
                test_dataset = FolderDataset(
                    test_stats.benign_files,
                    test_stats.malignant_files,
                    transform
                )
                metrics = evaluate_model(model, test_dataset)

            # Store results for comparison
            if "batch_results" not in st.session_state:
                st.session_state.batch_results = {}
            st.session_state.batch_results[weights_path] = {
                "metrics": metrics,
                "model_name": model_info.name,
            }

            st.success("Evaluation complete!")

        # Display results if available
        if "batch_results" in st.session_state and weights_path in st.session_state.batch_results:
            result = st.session_state.batch_results[weights_path]
            metrics = result["metrics"]

            metric_cols = st.columns(4)
            metric_cols[0].metric("AUC", f"{metrics['auc']:.3f}")
            metric_cols[1].metric("Sensitivity", f"{metrics['sensitivity']:.1%}")
            metric_cols[2].metric("Specificity", f"{metrics['specificity']:.1%}")
            metric_cols[3].metric("Accuracy", f"{metrics['accuracy']:.1%}")

            st.caption(f"Evaluated on {metrics['n_samples']} images "
                      f"({metrics['n_malignant']} malignant, {metrics['n_benign']} benign)")

            # Show comparison if multiple models have been evaluated
            if len(st.session_state.batch_results) > 1:
                st.divider()
                st.markdown("**Model Comparison**")
                comparison_data = []
                for _path, res in st.session_state.batch_results.items():
                    m = res["metrics"]
                    comparison_data.append({
                        "Model": res["model_name"],
                        "AUC": f"{m['auc']:.3f}",
                        "Sensitivity": f"{m['sensitivity']:.1%}",
                        "Specificity": f"{m['specificity']:.1%}",
                    })
                st.table(comparison_data)

        st.divider()
    else:
        st.info("💡 Set a test folder in **Project Overview** to enable batch evaluation.")
        st.divider()

    # Single Image Inference.
    st.subheader("Single Image Inference")
    st.markdown("Upload a mammogram image to get a malignancy prediction.")

    uploaded_file = st.file_uploader(
        "Upload mammogram",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        help="Supported formats: PNG, JPEG, DICOM"
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.lower().split('.')[-1]

        with st.spinner("Processing image..."):
            if file_ext in ["dcm", "dicom"]:
                img_array = load_dicom(uploaded_file.read())
            else:
                img = Image.open(uploaded_file).convert('RGB')
                img_array = np.array(img).astype(np.uint8)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Image")
                st.image(img_array, use_container_width=True)

            processed_img = preprocess_image(img_array, transform)
            result = run_inference(model, processed_img)

            with col2:
                st.subheader("Prediction")
                if result.classification == Classification.MALIGNANT:
                    st.error(f"**{result.classification.value}**")
                else:
                    st.success(f"**{result.classification.value}**")

                st.metric("Malignancy Probability", f"{result.malignant_prob:.1%}")
                st.progress(result.malignant_prob)

                # Add confidence explanation
                confidence_level, confidence_explanation = get_confidence_description(
                    result.malignant_prob
                )
                st.markdown(f"**Confidence:** {confidence_level}")
                st.caption(confidence_explanation)

                # Warn about potential out-of-distribution inputs
                if result.malignant_prob < 0.01 or result.malignant_prob > 0.99:
                    st.warning(
                        "**Extreme prediction detected.** "
                        "Please verify this is a valid mammogram image. "
                        "Non-mammography images may produce misleading results."
                    )


def finetune_tab():
    st.header("Fine-tune Model")
    st.markdown("""
    Adapt the model to your local imaging equipment by fine-tuning on your own labeled data.
    """)

    # Use train folder from Project Overview
    folder_path = st.session_state.get("train_folder", "")
    stats = st.session_state.get("train_stats", None)

    if not folder_path or not stats:
        st.info("💡 Set a training folder in **Project Overview** first.")
        return

    st.markdown(f"**Training folder:** `{folder_path}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Benign", stats.benign)
    col2.metric("Malignant", stats.malignant)
    col3.metric("Total", stats.total)

    if stats.total < 10:
        st.warning("Consider adding more images for better fine-tuning results (recommended: 50+)")

    st.divider()

    preset = st.selectbox(
        "Training duration",
        options=list(TRAINING_PRESETS.keys()),
        index=1,
        help="Longer training generally produces better results"
    )

    base_weights = st.text_input(
        "Base model weights",
        value=DEFAULT_WEIGHTS,
        help="Pre-trained weights to fine-tune from"
    )

    output_name = st.text_input(
        "Output model name",
        value="my-finetuned-model",
        help="Name for the fine-tuned model"
    )

    if st.button("Start Fine-tuning", type="primary", use_container_width=True):
        if not Path(base_weights).exists():
            st.error(f"Base weights not found: {base_weights}")
            return

        st.session_state.training_active = True

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        with st.spinner("Initializing..."):
            model = load_model(base_weights)

            # Stratified train/val split (80/20)
            benign_files = list(stats.benign_files)
            malignant_files = list(stats.malignant_files)
            train_benign, train_malignant, val_benign, val_malignant = \
                stratified_train_val_split(benign_files, malignant_files, val_fraction=0.2)

            train_transform = get_train_transform()
            val_transform = get_inference_transform()
            train_dataset = FolderDataset(train_benign, train_malignant, train_transform)
            val_dataset = FolderDataset(val_benign, val_malignant, val_transform)

            n_train = len(train_dataset)
            n_val = len(val_dataset)
            status_text.markdown(f"Training on {n_train} images, validating on {n_val} images")

        preset_config = TRAINING_PRESETS[preset]
        start_time = time.time()

        def update_progress(progress, epoch, total_epochs, stage, loss, acc):
            progress_bar.progress(progress)
            elapsed = time.time() - start_time
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                remaining_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                remaining_str = "calculating..."

            status_text.markdown(f"**{stage}** - Epoch {epoch}/{total_epochs}")
            with metrics_container.container():
                cols = st.columns(4)
                cols[0].metric("Progress", f"{progress:.0%}")
                cols[1].metric("Loss", f"{loss:.4f}")
                cols[2].metric("Accuracy", f"{acc:.1%}")
                cols[3].metric("Remaining", remaining_str)

        model = run_finetuning(
            model, train_dataset,
            epochs=preset_config["epochs"],
            stage1_epochs=preset_config["stage1_epochs"],
            progress_callback=update_progress
        )

        # Save model
        output_dir = Path("checkpoints/user") / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "best_model.safetensors"
        model.save_weights(str(output_path))

        elapsed = time.time() - start_time

        # Evaluate on validation set
        status_text.markdown("**Evaluating on validation set...**")
        val_metrics = evaluate_model(model, val_dataset)

        # Clear progress display
        progress_bar.empty()
        metrics_container.empty()
        status_text.empty()

        # Display results
        st.divider()
        st.subheader("Fine-tuning Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Time", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")
            st.metric("Model Saved To", str(output_path))

        with col2:
            st.metric("Validation AUC", f"{val_metrics['auc']:.3f}")
            st.metric("Sensitivity", f"{val_metrics['sensitivity']:.1%}")
            st.metric("Specificity", f"{val_metrics['specificity']:.1%}")

        st.caption(
            f"Validated on {val_metrics['n_samples']} images "
            f"({val_metrics['n_malignant']} malignant, {val_metrics['n_benign']} benign)")

        # Result interpretation and guidance
        st.divider()
        auc = val_metrics['auc']

        if auc >= 0.80:
            st.success(
                "**Good results.** The fine-tuned model shows strong performance on your data.")
            st.markdown("""
            The model is ready for use. You can now:
            - Switch to this model in the Project Overview tab
            - Use it for inference on new images
            """)
        elif auc >= 0.70:
            st.info("**Reasonable results.** The model shows moderate performance.")
            st.markdown("""
            **Suggestions to improve:**
            - Add more training images (especially for the minority class)
            - Try the **Thorough** preset for longer training
            - Ensure consistent image quality across your dataset
            """)
        elif auc >= 0.60:
            cases_to_add = 'malignant' if stats.malignant < stats.benign else 'benign'
            st.warning("**Suboptimal results.** Performance is below typical clinical thresholds.")
            st.markdown(f"""
            **Possible causes:**
            - Too few training images (current: {stats.total}, recommended: 50+)
            - Class imbalance (benign: {stats.benign}, malignant: {stats.malignant})
            - Inconsistent image quality or labelling errors

            **Recommendations:**
            - Add more labelled images, especially {cases_to_add} cases
            - Try the **Thorough** preset
            - Review your labels for accuracy
            """)
        else:
            st.error("**Poor results.** The model is performing near random chance.")
            st.markdown(f"""
            **This may indicate:**
            - Insufficient training data (current: {stats.total} images)
            - Severe class imbalance
            - Data quality issues or labelling errors
            - The base model may not be suitable for your imaging equipment

            **Recommendations:**
            - Significantly increase your dataset size (aim for 100+ images)
            - Balance your classes (similar numbers of benign and malignant)
            - Verify all labels are correct
            - Try training with the **Thorough** preset
            """)

        st.divider()
        if st.button("Use this model for inference", type="primary"):
            st.session_state.current_weights = str(output_path)
            st.session_state.pop("model", None)
            st.rerun()


def project_overview_tab():
    st.header("Project Overview")
    st.markdown("""
    Configure your model and datasets here before running inference or fine-tuning.
    """)

    current_weights = st.session_state.get("current_weights", DEFAULT_WEIGHTS)

    # Model Selection.
    st.subheader("1. Select Model")

    checkpoints_dir = Path("checkpoints")
    available_models = []
    model_options = []

    if checkpoints_dir.exists():
        # Collect all models with their metadata
        all_models = []
        for subdir in ["default", "user"]:
            subdir_path = checkpoints_dir / subdir
            if not subdir_path.exists():
                continue
            for model_dir in subdir_path.iterdir():
                if model_dir.is_dir():
                    weights_file = model_dir / "best_model.safetensors"
                    if weights_file.exists():
                        weights_path = str(weights_file)
                        model_info = get_model_display_info(weights_path)
                        model_name = model_dir.name
                        is_default = MODEL_DESCRIPTIONS.get(model_name, {}).get("is_default", False)
                        all_models.append((weights_path, model_info, is_default))

        # Sort default model, then vendor models, then user models (alphabetically within groups)
        all_models.sort(key=lambda x: (not x[2], not x[1].is_vendor, x[1].name.lower()))

        for weights_path, model_info, _is_default in all_models:
            available_models.append(weights_path)
            prefix = VENDOR_MODEL_PREFIX if model_info.is_vendor else USER_MODEL_PREFIX
            model_options.append(f"{prefix}{model_info.name}")

    if available_models:
        # Find current selection index
        current_index = 0
        if current_weights in available_models:
            current_index = available_models.index(current_weights)

        help_text = (
            f"{VENDOR_MODEL_PREFIX}= Vendor-provided model, "
            f"{USER_MODEL_PREFIX}= User fine-tuned model")
        selected_index = st.selectbox(
            "Select model",
            options=range(len(model_options)),
            format_func=lambda i: model_options[i],
            index=current_index,
            help=help_text
        )

        selected_path = available_models[selected_index]
        model_info = get_model_display_info(selected_path)

        st.caption(f"*{model_info.description}*")

        if selected_path != current_weights and st.button("Load selected model", type="primary"):
            st.session_state.current_weights = selected_path
            st.session_state.pop("model", None)
            st.success(f"Switched to: {model_info.name}")
            st.rerun()
    else:
        st.warning("No checkpoints found")

    st.divider()

    # Dataset Selection.
    st.subheader("2. Configure Datasets (Optional)")
    st.markdown("""
    Set your training and test data folders here for batch inference and fine-tuning.

    **Required folder structure:**
    ```
    your_folder/
    ├── benign/
    │   ├── image1.png
    │   └── ...
    └── malignant/
        ├── image1.png
        └── ...
    ```
    """)

    col1, col2 = st.columns(2)

    with col1:
        train_folder = st.text_input(
            "Training data folder",
            value=st.session_state.get("train_folder", ""),
            placeholder="/path/to/train_data",
            help="Folder with benign/ and malignant/ subfolders for fine-tuning"
        )
        if train_folder:
            result = validate_training_folder(train_folder)
            if result.error:
                st.error(result.error)
            else:
                image_count = (
                    f"✓ {result.total} images ({result.benign} benign, "
                    f"{result.malignant} malignant)"
                )
                st.success(image_count)
                st.session_state.train_folder = train_folder
                st.session_state.train_stats = result

    with col2:
        test_folder = st.text_input(
            "Test data folder",
            value=st.session_state.get("test_folder", ""),
            placeholder="/path/to/test_data",
            help="Folder with benign/ and malignant/ subfolders for evaluation"
        )
        if test_folder:
            result = validate_training_folder(test_folder)
            if result.error:
                st.error(result.error)
            else:
                success_message = (
                    f"✓ {result.total} images ({result.benign} benign, "
                    f"{result.malignant} malignant)"
                )
                st.success(success_message)
                st.session_state.test_folder = test_folder
                st.session_state.test_stats = result

    st.divider()

    # Current Configuration Summary.
    st.subheader("Current Configuration")
    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.markdown("**Model**")
        model_info = get_model_display_info(current_weights)
        st.code(model_info.name)

    with config_col2:
        st.markdown("**Settings**")
        st.code(f"Image size: {DEFAULT_WIDTH} x {DEFAULT_HEIGHT}\nThreshold: 0.5")


def main():
    st.set_page_config(
        page_title="Mammogram Classifier",
        page_icon="🩺",
        layout="centered"
    )

    # Load custom CSS for accessibility fixes
    css_file = Path(__file__).parent.parent / ".streamlit" / "style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

    st.title("🩺 Mammogram Classifier")

    if "current_weights" not in st.session_state:
        st.session_state.current_weights = DEFAULT_WEIGHTS

    tab1, tab2, tab3 = st.tabs(["Project Overview", "Inference", "Fine-tune"])

    with tab1:
        project_overview_tab()

    with tab2:
        inference_tab()

    with tab3:
        finetune_tab()


if __name__ == "__main__":
    main()
