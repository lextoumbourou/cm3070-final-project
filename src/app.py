"""Streamlit web interface for mammogram classification."""

from pathlib import Path
import sys

import streamlit as st
import numpy as np
from PIL import Image
import pydicom
import mlx.core as mx
import albumentations as A

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "mlx-image" / "src"))
from src.models.whole_image_classifier import create_whole_image_classifier

DEFAULT_WEIGHTS = "checkpoints/cbis-whole-v1/best_model.safetensors"
TARGET_HEIGHT = 896
TARGET_WIDTH = 1152


@st.cache_resource
def load_model(weights_path: str):
    """Load model with caching to avoid reloading on each interaction."""
    model = create_whole_image_classifier(
        patch_weights_path=None,
        backbone_name="resnet50",
        num_classes=2
    )
    model.load_weights(weights_path)
    model.eval()
    return model


def get_transform():
    return A.Compose([
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ])


def load_dicom(file_bytes) -> np.ndarray:
    """Load DICOM file and convert to RGB numpy array."""
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


def preprocess_image(img: np.ndarray, transform) -> mx.array:
    """Preprocess image for model input."""
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)


def run_inference(model, img: mx.array) -> tuple[float, str]:
    """Run inference and return probability and classification."""
    inputs = mx.expand_dims(img, 0)
    logits = model(inputs)
    probs = mx.softmax(logits, axis=1)
    mx.eval(probs)

    malignant_prob = probs[0, 1].item()
    classification = "Malignant" if malignant_prob >= 0.5 else "Benign"

    return malignant_prob, classification


def main():
    st.set_page_config(
        page_title="Mammogram Classifier",
        page_icon="🔬",
        layout="centered"
    )

    st.title("Mammogram Classification")
    st.markdown("Upload a mammogram image to get a malignancy prediction.")

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input(
            "Model weights path",
            value=DEFAULT_WEIGHTS,
            help="Path to the trained model weights"
        )

        if not Path(weights_path).exists():
            st.error(f"Weights file not found: {weights_path}")
            st.stop()

    model = load_model(weights_path)
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

                st.caption(
                    f"Image resized to {TARGET_WIDTH}x{TARGET_HEIGHT} for inference. "
                    f"Threshold: 0.5"
                )


if __name__ == "__main__":
    main()
