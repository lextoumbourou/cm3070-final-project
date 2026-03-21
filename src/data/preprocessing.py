"""Mammogram image preprocessing utilities."""

import cv2
import numpy as np


def normalise_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalise image to 0-255 uint8 range."""
    if img.dtype == np.uint8:
        return img

    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def apply_morphological_transforms(
    mask: np.ndarray,
    kernel_size: int = 100,
    iterations: int = 2
) -> np.ndarray:
    """Apply morphological opening then closing to clean up binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def get_largest_contour_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find largest contour in mask and return its bounding box (x, y, w, h)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0])

    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def get_breast_bbox(
    img: np.ndarray,
    blur_kernel: tuple[int, int] = (5, 5),
    morph_kernel_size: int = 100,
    morph_iterations: int = 2
) -> tuple[int, int, int, int]:
    """
    Get bounding box for breast region using Otsu thresholding.

    Returns (x, y, w, h) of the breast bounding box.
    """
    img_norm = normalise_to_uint8(img)
    blurred = cv2.GaussianBlur(img_norm, blur_kernel, 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = apply_morphological_transforms(mask, morph_kernel_size, morph_iterations)
    return get_largest_contour_bbox(cleaned)


def crop_to_breast(img: np.ndarray, **kwargs) -> np.ndarray:
    """Crop image to breast region."""
    x, y, w, h = get_breast_bbox(img, **kwargs)
    return img[y:y+h, x:x+w]
