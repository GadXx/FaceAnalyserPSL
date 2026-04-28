from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.data.io import load_image, save_image, write_json
from src.face_aesthetics.features.extraction import derive_semantic_landmark_bundle
from src.face_aesthetics.pipelines.preprocessing import preprocess_single_image


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    return value


def _to_color(group: str | None) -> tuple[int, int, int]:
    palette = {
        "jaw_contour": (78, 110, 180),
        "brows_eyes": (130, 90, 188),
        "eyes": (136, 95, 220),
        "nose": (199, 122, 59),
        "nose_mouth_core": (190, 142, 66),
        "mouth": (54, 86, 221),
        "chin_jaw": (81, 126, 204),
    }
    return palette.get(str(group or ""), (30, 210, 255))


def _draw_bbox(image: np.ndarray, bbox: list[float] | None) -> None:
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), (54, 189, 95), 2)


def _draw_raw_landmarks(image: np.ndarray, raw_landmarks: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    scale = max(0.3, min(0.6, image.shape[1] / 1200.0))
    for point in raw_landmarks:
        index = int(point.get("index", -1))
        x = int(round(float(point.get("x", 0.0))))
        y = int(round(float(point.get("y", 0.0))))
        color = _to_color(point.get("group"))
        cv2.circle(canvas, (x, y), 2, color, -1)
        cv2.putText(
            canvas,
            str(index),
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_semantic_points(
    image: np.ndarray,
    *,
    bbox: list[float] | None,
    semantic_points: dict[str, dict[str, float]],
    point_sources: dict[str, list[int]] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    _draw_bbox(canvas, bbox)
    point_sources = point_sources or {}
    scale = max(0.35, min(0.7, image.shape[1] / 1000.0))
    for name, payload in semantic_points.items():
        x = int(round(float(payload["x"])))
        y = int(round(float(payload["y"])))
        color = _to_color("mouth" if "mouth" in name or "lip" in name else ("nose" if "nose" in name else None))
        cv2.circle(canvas, (x, y), 3, color, -1)
        suffix = ""
        source_indices = point_sources.get(name, [])
        if source_indices:
            suffix = f" [{','.join(str(v) for v in source_indices)}]"
        cv2.putText(
            canvas,
            f"{name}{suffix}",
            (x + 4, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug raw InsightFace 106 landmarks and semantic morphometric points "
            "for a single image."
        )
    )
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "debug")
    parser.add_argument("--preferred-execution-provider", type=str, default="CPUExecutionProvider")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--strict-gpu", type=_str_to_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_image_path = output_dir / "analysis_image.jpg"
    aligned_image_path = output_dir / "aligned_face.jpg"
    artifacts = preprocess_single_image(
        image_path=args.image_path.resolve(),
        project_root=PROJECT_ROOT,
        output_analysis_path=analysis_image_path,
        output_aligned_path=aligned_image_path,
        overrides={
            "preferred_execution_provider": args.preferred_execution_provider,
            "fallback_execution_provider": "CPUExecutionProvider",
            "device_id": args.device_id,
            "strict_gpu_mode": args.strict_gpu,
        },
    )
    row = dict(artifacts.row)
    bbox = _parse_json_field(row.get("bbox"))
    landmarks = _parse_json_field(row.get("landmarks"))
    landmarks_dense = _parse_json_field(row.get("landmarks_dense"))
    semantic_bundle = derive_semantic_landmark_bundle(
        landmarks=landmarks,
        landmarks_dense=landmarks_dense,
        bbox=bbox,
    )

    render_image_path = (
        artifacts.measurement_image_path
        or artifacts.analysis_image_path
        or artifacts.aligned_image_path
        or args.image_path.resolve()
    )
    image = load_image(render_image_path)
    raw_landmarks = list(semantic_bundle.get("raw_landmarks", []))
    semantic_points = dict(semantic_bundle.get("semantic_points", {}))
    semantic_points_debug = list(semantic_bundle.get("semantic_points_debug", []))
    source_map = {
        str(entry.get("point_name")): [int(value) for value in entry.get("source_indices", [])]
        for entry in semantic_points_debug
        if isinstance(entry, dict)
    }

    raw_overlay = _draw_raw_landmarks(image, raw_landmarks)
    semantic_overlay = _draw_semantic_points(
        image,
        bbox=bbox if isinstance(bbox, list) else None,
        semantic_points=semantic_points,
    )
    semantic_with_sources_overlay = _draw_semantic_points(
        image,
        bbox=bbox if isinstance(bbox, list) else None,
        semantic_points=semantic_points,
        point_sources=source_map,
    )

    raw_overlay_path = output_dir / "landmarks_raw_106_indices.png"
    semantic_overlay_path = output_dir / "landmarks_semantic_points.png"
    semantic_with_sources_path = output_dir / "landmarks_semantic_sources.png"
    save_image(raw_overlay_path, raw_overlay)
    save_image(semantic_overlay_path, semantic_overlay)
    save_image(semantic_with_sources_path, semantic_with_sources_overlay)

    debug_payload = {
        "image_path": str(args.image_path.resolve()),
        "analysis_image_path": str(render_image_path.resolve()),
        "analysis_image_role": artifacts.measurement_image_role,
        "processing_status": row.get("processing_status"),
        "quality_pass": row.get("quality_pass"),
        "quality_fail_reason": row.get("quality_fail_reason"),
        "bbox": bbox,
        "raw_landmarks": raw_landmarks,
        "raw_landmarks_debug": semantic_bundle.get("raw_landmarks_debug", []),
        "semantic_points": semantic_points,
        "semantic_points_debug": semantic_points_debug,
        "semantic_validation": semantic_bundle.get("semantic_validation"),
        "artifacts": {
            "landmarks_raw_106_indices": str(raw_overlay_path.resolve()),
            "landmarks_semantic_points": str(semantic_overlay_path.resolve()),
            "landmarks_semantic_sources": str(semantic_with_sources_path.resolve()),
            "analysis_image": str(render_image_path.resolve()),
        },
    }
    debug_json_path = output_dir / "landmarks_debug.json"
    write_json(debug_payload, debug_json_path)

    print(f"analysis_image={render_image_path.resolve()}")
    print(f"landmarks_raw_106_indices={raw_overlay_path.resolve()}")
    print(f"landmarks_semantic_points={semantic_overlay_path.resolve()}")
    print(f"landmarks_semantic_sources={semantic_with_sources_path.resolve()}")
    print(f"landmarks_debug_json={debug_json_path.resolve()}")


if __name__ == "__main__":
    main()
