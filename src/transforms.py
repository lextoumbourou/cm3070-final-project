"""Shared image transforms for mammography classification.

Augmentation strategy follows Shen et al. (2019):
"Deep Learning to Improve Breast Cancer Detection on Screening Mammography"
"""

import albumentations as A
import cv2
import mlx.core as mx
import numpy as np

# Default dimensions for whole image classification
DEFAULT_HEIGHT = 896
DEFAULT_WIDTH = 1152


def get_inference_transform(
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> A.Compose:
    """Get transform for inference/validation (resize only)."""
    return A.Compose([A.Resize(height=height, width=width)])


def get_train_transform(
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> A.Compose:
    """Get transform for training with augmentations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.2, p=0.5),
        A.Resize(height=height, width=width),
    ])


def get_tta_transforms(
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> list[A.Compose]:
    """Get test-time augmentation transforms.

    Returns 4 variants: original, h-flip, v-flip, both flips.
    Following Shen et al. 2019: "horizontally and vertically flipping
    an image to obtain four images and taking an average of the four
    images' scores"
    """
    base_resize = A.Resize(height=height, width=width)
    return [
        A.Compose([base_resize]),
        A.Compose([base_resize, A.HorizontalFlip(p=1.0)]),
        A.Compose([base_resize, A.VerticalFlip(p=1.0)]),
        A.Compose([base_resize, A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
    ]


def preprocess_image(img: np.ndarray, transform: A.Compose) -> mx.array:
    """Apply transform and convert to MLX array for inference.

    Args:
        img: Input image as numpy array (H, W, C) uint8
        transform: Albumentations transform to apply

    Returns:
        MLX array ready for model input, normalized to [0, 1]
    """
    img = transform(image=img)['image']
    img = img.astype(np.float32) / 255.0
    return mx.array(img)
