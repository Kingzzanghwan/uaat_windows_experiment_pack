from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _to_float_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = image.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def image_quality_features(image: Image.Image | np.ndarray) -> dict[str, float]:
    """Small, dependency-light image quality/context features.

    Returned values are intentionally simple and stable:
    - brightness: average pixel intensity, 0 to 1
    - contrast: standard deviation of grayscale intensity, 0 to 1
    - blur: variance of a Laplacian-like edge image, clipped/log-scaled to 0 to 1
    - saturation: average max-min channel gap, 0 to 1
    """
    arr = _to_float_rgb(image)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())

    # Laplacian-like filter without scipy/opencv.
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    blur_raw = float(np.var(lap))
    # Lower edge variance means blurrier. We store blur_risk: 1 means blurry.
    edge_strength = np.log1p(500.0 * blur_raw) / np.log1p(500.0)
    edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
    blur_risk = float(1.0 - edge_strength)

    return {
        "ctx_brightness": brightness,
        "ctx_contrast": contrast,
        "ctx_blur_risk": blur_risk,
        "ctx_saturation": saturation,
    }


def image_quality_from_path(path: str | Path) -> dict[str, float]:
    with Image.open(path) as im:
        return image_quality_features(im)


def batch_quality_from_uint8(images: np.ndarray) -> list[dict[str, float]]:
    """Compute quality features for a batch of HWC uint8 images."""
    return [image_quality_features(img) for img in images]
