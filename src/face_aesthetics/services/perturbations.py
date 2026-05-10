from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.face_aesthetics.data.io import load_image, save_image


def _crop_shift(image: np.ndarray, shift_x_fraction: float, shift_y_fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    shift_x = int(round(width * shift_x_fraction))
    shift_y = int(round(height * shift_y_fraction))
    source = np.float32(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
        ]
    )
    destination = np.float32(
        [
            [float(shift_x), float(shift_y)],
            [float(width - 1 + shift_x), float(shift_y)],
            [float(shift_x), float(height - 1 + shift_y)],
        ]
    )
    matrix = cv2.getAffineTransform(source, destination)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR)


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise ValueError("Failed to encode perturbation image as JPEG.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode JPEG perturbation image.")
    return decoded


def _blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.0)


def _brightness(image: np.ndarray, delta: float) -> np.ndarray:
    adjusted = image.astype(np.float32) + float(delta)
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_named_perturbation(image: np.ndarray, perturbation_name: str, config: dict[str, Any]) -> np.ndarray:
    if perturbation_name == "crop_shift":
        return _crop_shift(image, float(config["crop_shift_x"]), float(config["crop_shift_y"]))
    if perturbation_name == "resize":
        return _resize(image, float(config["resize_scale"]))
    if perturbation_name == "jpeg_compression":
        return _jpeg_compress(image, int(config["jpeg_quality"]))
    if perturbation_name == "mild_blur":
        return _blur(image, int(config["blur_kernel_size"]))
    if perturbation_name == "brightness_change":
        return _brightness(image, float(config["brightness_delta"]))
    if perturbation_name == "small_rotation":
        return _rotate(image, float(config["rotation_deg"]))
    raise ValueError(f"Unsupported perturbation: {perturbation_name}")


def generate_perturbation_artifacts(
    *,
    image_path: Path,
    output_dir: Path,
    perturbation_config: dict[str, Any],
) -> list[dict[str, str]]:
    image = load_image(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    perturbations = [
        "crop_shift",
        "resize",
        "jpeg_compression",
        "mild_blur",
        "brightness_change",
        "small_rotation",
    ]
    artifacts: list[dict[str, str]] = []
    for perturbation_name in perturbations:
        perturbed = apply_named_perturbation(image, perturbation_name, perturbation_config)
        output_path = output_dir / f"{image_path.stem}__{perturbation_name}.jpg"
        save_image(output_path, perturbed)
        artifacts.append(
            {
                "perturbation_name": perturbation_name,
                "image_path": str(output_path.resolve()),
            }
        )
    return artifacts
