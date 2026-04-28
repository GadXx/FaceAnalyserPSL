from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_TEMPLATE_SHAPE = (5, 3)
DEFAULT_CANONICAL_TEMPLATE_5PT = np.asarray(
    [
        [38.2946, 51.6963, 0.0],
        [73.5318, 51.5014, 0.0],
        [56.0252, 71.7366, 8.0],
        [41.5493, 92.3655, 2.5],
        [70.7299, 92.2041, 2.5],
    ],
    dtype=np.float64,
)


@dataclass
class ThreeDAvailability:
    enabled: bool
    active: bool
    status: str
    reason: str | None
    template_path: str | None


def create_default_canonical_face_template(template_path: Path) -> Path:
    template_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(template_path, DEFAULT_CANONICAL_TEMPLATE_5PT)
    return template_path


def ensure_default_canonical_face_template(
    *,
    template_path: Path,
    auto_create: bool,
) -> Path:
    if template_path.exists() or not auto_create:
        return template_path
    if template_path.name != "canonical_face_5pt.npy":
        return template_path
    return create_default_canonical_face_template(template_path)


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    return _rotation_matrix_z(roll_deg) @ _rotation_matrix_y(yaw_deg) @ _rotation_matrix_x(pitch_deg)


class OptionalFace3DProcessor:
    def __init__(self, *, enabled: bool, template_path: Path) -> None:
        self.template_path = template_path
        self._template_points: np.ndarray | None = None

        if not enabled:
            self.availability = ThreeDAvailability(
                enabled=False,
                active=False,
                status="disabled",
                reason="3d_branch_disabled",
                template_path=str(template_path),
            )
            return

        if not template_path.exists():
            self.availability = ThreeDAvailability(
                enabled=True,
                active=False,
                status="skipped_missing_models",
                reason=f"missing_template:{template_path}",
                template_path=str(template_path),
            )
            return

        template = np.load(template_path)
        if tuple(template.shape) != REQUIRED_TEMPLATE_SHAPE:
            self.availability = ThreeDAvailability(
                enabled=True,
                active=False,
                status="skipped_invalid_template",
                reason=f"expected_shape={REQUIRED_TEMPLATE_SHAPE}, actual_shape={tuple(template.shape)}",
                template_path=str(template_path),
            )
            return

        self._template_points = np.asarray(template, dtype=np.float64)
        self.availability = ThreeDAvailability(
            enabled=True,
            active=True,
            status="active",
            reason=None,
            template_path=str(template_path),
        )

    def availability_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.availability.enabled,
            "active": self.availability.active,
            "status": self.availability.status,
            "reason": self.availability.reason,
            "template_path": self.availability.template_path,
        }

    def build_artifact(
        self,
        *,
        image_id: str,
        landmarks_2d: list[list[float]] | None,
        yaw: float | None,
        pitch: float | None,
        roll: float | None,
    ) -> dict[str, Any]:
        if not self.availability.active or self._template_points is None:
            return {
                "image_id": image_id,
                "status": self.availability.status,
                "reason": self.availability.reason,
                "artifact_version": "template_pose_lift@1",
                "landmarks_3d": None,
            }

        if not landmarks_2d or len(landmarks_2d) < 5:
            return {
                "image_id": image_id,
                "status": "skipped_missing_landmarks",
                "reason": "landmarks_2d_missing_or_short",
                "artifact_version": "template_pose_lift@1",
                "landmarks_3d": None,
            }

        observed = np.asarray(landmarks_2d[:5], dtype=np.float64)
        template = self._template_points.copy()

        observed_eye_distance = float(np.linalg.norm(observed[1] - observed[0]))
        template_eye_distance = float(np.linalg.norm(template[1, :2] - template[0, :2]))
        if observed_eye_distance <= 0.0 or template_eye_distance <= 0.0:
            return {
                "image_id": image_id,
                "status": "skipped_invalid_geometry",
                "reason": "non_positive_eye_distance",
                "artifact_version": "template_pose_lift@1",
                "landmarks_3d": None,
            }

        template_centered = template - template[2]
        rotation = _rotation_matrix(float(yaw or 0.0), float(pitch or 0.0), float(roll or 0.0))
        rotated = template_centered @ rotation.T
        scale = observed_eye_distance / template_eye_distance
        scaled = rotated * scale
        translated = scaled.copy()
        translated[:, 0] += observed[2, 0]
        translated[:, 1] += observed[2, 1]

        return {
            "image_id": image_id,
            "status": "success",
            "reason": None,
            "artifact_version": "template_pose_lift@1",
            "template_path": str(self.template_path),
            "scale": float(scale),
            "pose": {
                "yaw": float(yaw) if yaw is not None else None,
                "pitch": float(pitch) if pitch is not None else None,
                "roll": float(roll) if roll is not None else None,
            },
            "landmarks_3d": translated.tolist(),
        }


def save_3d_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
