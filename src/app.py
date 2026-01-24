"""Basic Streamlit web interface for fine-tuning."""

from pathlib import Path
import sys
import time
import csv

import streamlit as st
import numpy as np
from PIL import Image
import pydicom
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import albumentations as A
import cv2

# Ensure mlx-image is in path.
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlx-image" / "src"))
from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from src.models.whole_image_classifier import create_whole_image_classifier

DEFAULT_WEIGHTS = "checkpoints/cbis-whole-wd-only/best_model.safetensors"
TARGET_HEIGHT = 896
TARGET_WIDTH = 1152

TRAINING_PRESETS = {
    "Quick": {"epochs": 5, "stage1_epochs": 2},
    "Standard": {"epochs": 15, "stage1_epochs": 5},
    "Thorough": {"epochs": 30, "stage1_epochs": 10},
}


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


def get_transform():
    return A.Compose([A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH)])


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.2, p=0.5),
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ])


def load_dicom(file_bytes):
    import tempfile
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


def preprocess_image(img, transform):
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)


def run_inference(model, img):
    inputs = mx.expand_dims(img, 0)
    logits = model(inputs)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)
    malignant_prob = probs[0, 1].item()
    classification = "Malignant" if malignant_prob >= 0.5 else "Benign"
    return malignant_prob, classification


def validate_training_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return None, "Folder does not exist"

    benign_folder = folder / "benign"
    malignant_folder = folder / "malignant"

    if not benign_folder.exists() or not malignant_folder.exists():
        return None, "Folder must contain 'benign/' and 'malignant/' subfolders"

    image_extensions = {'.png', '.jpg', '.jpeg', '.dcm', '.dicom'}
    benign_images = [f for f in benign_folder.iterdir() if f.suffix.lower() in image_extensions]
    malignant_images = [f for f in malignant_folder.iterdir() if f.suffix.lower() in image_extensions]

    if len(benign_images) == 0 or len(malignant_images) == 0:
        return None, "Both folders must contain at least one image"

    return {
        "benign": len(benign_images),
        "malignant": len(malignant_images),
        "total": len(benign_images) + len(malignant_images),
        "benign_files": benign_images,
        "malignant_files": malignant_images
    }, None


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
    st.markdown("Upload a mammogram image to get a malignancy prediction.")

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
    transform = get_transform()

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
            malignant_prob, classification = run_inference(model, processed_img)

            with col2:
                st.subheader("Prediction")
                if classification == "Malignant":
                    st.error(f"**{classification}**")
                else:
                    st.success(f"**{classification}**")
                st.metric("Malignancy Probability", f"{malignant_prob:.1%}")
                st.progress(malignant_prob)


def finetune_tab():
    st.header("Fine-tune Model")
    st.markdown("""
    Adapt the model to your local imaging equipment by fine-tuning on your own labeled data.

    **Prepare your data:**
    1. Create a folder with two subfolders: `benign/` and `malignant/`
    2. Place your mammogram images in the appropriate subfolder
    3. Select the folder below and start fine-tuning
    """)

    folder_path = st.text_input(
        "Training data folder",
        placeholder="/path/to/your/training_data",
        help="Folder containing 'benign/' and 'malignant/' subfolders with images"
    )

    if folder_path:
        stats, error = validate_training_folder(folder_path)

        if error:
            st.error(error)
        else:
            st.success("Folder structure validated")

            col1, col2, col3 = st.columns(3)
            col1.metric("Benign", stats["benign"])
            col2.metric("Malignant", stats["malignant"])
            col3.metric("Total", stats["total"])

            if stats["total"] < 10:
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

                    all_files = stats["benign_files"] + stats["malignant_files"]
                    np.random.shuffle(all_files)

                    benign_files = [f for f in all_files if f.parent.name == "benign"]
                    malignant_files = [f for f in all_files if f.parent.name == "malignant"]

                    train_transform = get_train_transform()
                    dataset = FolderDataset(benign_files, malignant_files, train_transform)

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
                    model, dataset,
                    epochs=preset_config["epochs"],
                    stage1_epochs=preset_config["stage1_epochs"],
                    progress_callback=update_progress
                )

                output_dir = Path("checkpoints") / output_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "best_model.safetensors"
                model.save_weights(str(output_path))

                elapsed = time.time() - start_time
                st.success(f"Fine-tuning complete in {int(elapsed // 60)}m {int(elapsed % 60)}s")
                st.info(f"Model saved to: `{output_path}`")

                if st.button("Use this model for inference"):
                    st.session_state.current_weights = str(output_path)
                    st.session_state.pop("model", None)
                    st.rerun()


def settings_tab():
    st.header("Settings")

    current_weights = st.session_state.get("current_weights", DEFAULT_WEIGHTS)

    st.subheader("Model Selection")

    checkpoints_dir = Path("checkpoints")
    available_models = []
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                weights_file = model_dir / "best_model.safetensors"
                if weights_file.exists():
                    available_models.append(str(weights_file))

    if available_models:
        selected = st.selectbox(
            "Select model",
            options=available_models,
            index=available_models.index(current_weights) if current_weights in available_models else 0
        )

        if selected != current_weights:
            if st.button("Load selected model"):
                st.session_state.current_weights = selected
                st.session_state.pop("model", None)
                st.success(f"Switched to: {selected}")
                st.rerun()
    else:
        st.warning("No models found in checkpoints/")

    st.divider()

    st.subheader("Manual weights path")
    manual_path = st.text_input("Weights file path", value=current_weights)
    if manual_path != current_weights and Path(manual_path).exists():
        if st.button("Load manual weights"):
            st.session_state.current_weights = manual_path
            st.session_state.pop("model", None)
            st.rerun()

    st.divider()

    st.subheader("Current Configuration")
    st.code(f"""
Active model: {current_weights}
Image size: {TARGET_WIDTH} x {TARGET_HEIGHT}
Threshold: 0.5
    """)


def main():
    st.set_page_config(
        page_title="Mammogram Classifier",
        page_icon="🩺",
        layout="centered"
    )

    st.title("🩺 Mammogram Classifier")

    if "current_weights" not in st.session_state:
        st.session_state.current_weights = DEFAULT_WEIGHTS

    tab1, tab2, tab3 = st.tabs(["Inference", "Fine-tune", "Settings"])

    with tab1:
        inference_tab()

    with tab2:
        finetune_tab()

    with tab3:
        settings_tab()


if __name__ == "__main__":
    main()
