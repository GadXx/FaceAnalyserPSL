from __future__ import annotations

import unittest

import numpy as np

from src.face_aesthetics.features.extraction import (
    derive_semantic_landmark_bundle,
    extract_angle_features,
    feature_audit_payload,
    feature_registry_rows,
    extract_normalized_measurements,
    extract_raw_measurements,
    extract_symmetry_features,
    validate_feature_values,
)


class MeasurementFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.landmarks = [
            [30.0, 40.0],
            [70.0, 40.0],
            [50.0, 60.0],
            [35.0, 80.0],
            [65.0, 80.0],
        ]
        self.bbox = [20.0, 20.0, 80.0, 100.0]
        self.landmarks_dense = [
            [26.0, 30.0], [32.0, 28.0], [38.0, 27.0], [62.0, 27.0], [68.0, 28.0], [74.0, 30.0],
            [32.0, 40.0], [38.0, 38.0], [46.0, 39.0], [54.0, 39.0], [62.0, 38.0], [68.0, 40.0],
            [49.0, 50.0], [51.0, 50.0], [45.0, 58.0], [55.0, 58.0],
            [36.0, 74.0], [42.0, 72.0], [50.0, 71.0], [58.0, 72.0], [64.0, 74.0],
            [40.0, 80.0], [50.0, 82.0], [60.0, 80.0],
            [46.0, 88.0], [54.0, 88.0], [50.0, 96.0],
        ]

    def test_raw_measurements_compute_expected_lengths(self) -> None:
        raw = extract_raw_measurements(self.landmarks, self.bbox, self.landmarks_dense)

        self.assertAlmostEqual(raw["raw_face_width_px"], 60.0)
        self.assertAlmostEqual(raw["raw_face_height_px"], 80.0)
        self.assertAlmostEqual(raw["raw_eye_distance_px"], 40.0)
        self.assertAlmostEqual(raw["raw_mouth_width_px"], 28.0)
        self.assertAlmostEqual(raw["raw_eye_to_nose_px"], 20.0)
        self.assertAlmostEqual(raw["raw_nose_to_mouth_px"], 14.0)
        self.assertAlmostEqual(raw["raw_nose_center_offset_px"], 0.0)
        self.assertAlmostEqual(raw["mouth_center_to_chin_height"], 22.0, places=4)
        self.assertAlmostEqual(raw["intercanthal_distance"], 8.0, places=4)
        self.assertAlmostEqual(raw["outercanthal_distance"], 36.0, places=4)
        self.assertAlmostEqual(raw["nose_base_width"], 10.0, places=4)

    def test_normalized_ratios_follow_documented_formulas(self) -> None:
        raw = extract_raw_measurements(self.landmarks, self.bbox, self.landmarks_dense)
        normalized = extract_normalized_measurements(raw)

        self.assertAlmostEqual(normalized["ratio_face_aspect"], 0.75)
        self.assertAlmostEqual(normalized["ratio_eye_distance_to_face_width"], 40.0 / 60.0)
        self.assertAlmostEqual(normalized["ratio_mouth_width_to_face_width"], 28.0 / 60.0)
        self.assertAlmostEqual(normalized["ratio_eye_to_nose_to_face_height"], 20.0 / 80.0)
        self.assertAlmostEqual(normalized["ratio_nose_center_offset_to_face_width"], 0.0)
        self.assertAlmostEqual(normalized["ratio_nose_width_to_face_width"], 10.0 / 60.0, places=4)

    def test_symmetry_metrics_reflect_balanced_landmarks(self) -> None:
        raw = extract_raw_measurements(self.landmarks, self.bbox, self.landmarks_dense)
        normalized = extract_normalized_measurements(raw)
        symmetry = extract_symmetry_features(raw, normalized)

        self.assertAlmostEqual(symmetry["sym_eye_nose_delta_ratio"], 0.0)
        self.assertAlmostEqual(symmetry["sym_mouth_nose_delta_ratio"], 0.0)
        self.assertAlmostEqual(symmetry["sym_horizontal_center_offset_ratio"], 0.0)
        self.assertAlmostEqual(symmetry["symmetry_index"], 1.0)

    def test_angle_metrics_include_landmark_lines_and_pose(self) -> None:
        angles = extract_angle_features(
            self.landmarks,
            {"yaw": 5.0, "pitch": -3.0, "roll": 2.0},
            self.landmarks_dense,
            self.bbox,
        )

        self.assertAlmostEqual(angles["angle_eye_line_deg"], 0.0)
        self.assertAlmostEqual(angles["angle_mouth_line_deg"], 0.0)
        self.assertAlmostEqual(angles["angle_eye_center_to_mouth_center_deg"], 90.0)
        self.assertAlmostEqual(angles["pose_yaw_deg"], 5.0)
        self.assertAlmostEqual(angles["pose_pitch_deg"], -3.0)
        self.assertAlmostEqual(angles["pose_roll_deg"], 2.0)
        self.assertAlmostEqual(angles["canthal_tilt_deg"], 0.0)

    def test_semantic_bundle_places_mouth_corners_around_mouth_center(self) -> None:
        bundle = derive_semantic_landmark_bundle(
            landmarks=self.landmarks,
            landmarks_dense=self.landmarks_dense,
            bbox=self.bbox,
        )
        semantic = bundle["semantic_points"]
        left = semantic["mouth_left"]
        right = semantic["mouth_right"]
        center = semantic["mouth_center"]
        nose = semantic["nose_tip"]

        self.assertLess(left["x"], center["x"])
        self.assertGreater(right["x"], center["x"])
        self.assertLess(left["x"], right["x"])
        self.assertGreater(left["y"], nose["y"])
        self.assertGreater(right["y"], nose["y"])
        self.assertAlmostEqual(center["x"], (left["x"] + right["x"]) / 2.0, places=4)

    def test_raw_landmark_debug_uses_analysis_coordinate_space_points(self) -> None:
        bundle = derive_semantic_landmark_bundle(
            landmarks=self.landmarks,
            landmarks_dense=self.landmarks_dense,
            bbox=self.bbox,
        )
        raw_debug = bundle["raw_landmarks_debug"]
        self.assertEqual(len(raw_debug), len(self.landmarks_dense))
        self.assertEqual(raw_debug[0]["source_type"], "raw")
        self.assertEqual(raw_debug[0]["source_indices"], [0])
        self.assertTrue(bundle["semantic_validation"]["passed"])

    def test_semantic_bundle_ignores_cheek_like_extremes_for_mouth_corners(self) -> None:
        dense_with_bad_extremes = list(self.landmarks_dense) + [
            [24.0, 78.0],
            [76.0, 78.0],
            [20.0, 84.0],
            [80.0, 84.0],
        ]
        bundle = derive_semantic_landmark_bundle(
            landmarks=self.landmarks,
            landmarks_dense=dense_with_bad_extremes,
            bbox=self.bbox,
        )
        semantic = bundle["semantic_points"]
        self.assertGreater(semantic["mouth_left"]["x"], 28.0)
        self.assertLess(semantic["mouth_right"]["x"], 72.0)
        self.assertLess(semantic["mouth_left"]["x"], semantic["mouth_right"]["x"])

    def test_semantic_bundle_prefers_canonical_106_indices_for_mouth_points(self) -> None:
        dense = np.tile(np.asarray([[50.0, 50.0]], dtype=np.float64), (106, 1))
        dense[33:43] = np.asarray([[36.0 + i, 40.0 + (i % 2)] for i in range(10)], dtype=np.float64)
        dense[87:97] = np.asarray([[56.0 + i, 40.0 + (i % 2)] for i in range(10)], dtype=np.float64)
        dense[52] = np.asarray([36.0, 78.0], dtype=np.float64)  # mouth_left
        dense[53] = np.asarray([50.0, 82.0], dtype=np.float64)  # lower_lip_bottom
        dense[61] = np.asarray([64.0, 78.0], dtype=np.float64)  # mouth_right
        dense[71] = np.asarray([50.0, 74.0], dtype=np.float64)  # upper_lip_top
        dense[78] = np.asarray([44.0, 66.0], dtype=np.float64)  # nose_base_left
        dense[84] = np.asarray([56.0, 66.0], dtype=np.float64)  # nose_base_right
        dense[86] = np.asarray([50.0, 60.0], dtype=np.float64)  # nose_tip

        bundle = derive_semantic_landmark_bundle(
            landmarks=[
                [30.0, 40.0],
                [70.0, 40.0],
                [50.0, 60.0],
                [18.0, 80.0],
                [82.0, 80.0],
            ],
            landmarks_dense=dense.tolist(),
            bbox=self.bbox,
        )
        semantic = bundle["semantic_points"]
        debug_by_name = {
            entry["point_name"]: entry
            for entry in bundle["semantic_points_debug"]
        }

        self.assertAlmostEqual(semantic["mouth_left"]["x"], 36.0, places=4)
        self.assertAlmostEqual(semantic["mouth_right"]["x"], 64.0, places=4)
        self.assertEqual(debug_by_name["mouth_left"]["source_type"], "insightface_106_index")
        self.assertEqual(debug_by_name["mouth_right"]["source_type"], "insightface_106_index")
        self.assertEqual(debug_by_name["mouth_left"]["source_indices"], [52])
        self.assertEqual(debug_by_name["mouth_right"]["source_indices"], [61])
        self.assertEqual(semantic["mouth_left"]["source_index_106"], 52)
        self.assertEqual(semantic["mouth_right"]["source_index_106"], 61)
        self.assertEqual(semantic["upper_lip_top"]["source_index_106"], 71)
        self.assertEqual(semantic["lower_lip_bottom"]["source_index_106"], 53)
        self.assertEqual(semantic["nose_tip"]["source_index_106"], 86)
        self.assertEqual(semantic["nose_base_left"]["source_index_106"], 78)
        self.assertEqual(semantic["nose_base_right"]["source_index_106"], 84)
        self.assertEqual(semantic["nose_base_left"]["source_region"], "nose")
        self.assertEqual(semantic["nose_base_right"]["source_region"], "nose")

    def test_validation_rejects_non_finite_and_impossible_ranges(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_values(
                {
                    "ratio_eye_distance_to_face_width": float("nan"),
                    "angle_eye_line_deg": 270.0,
                },
                image_id="sample-001",
            )

    def test_feature_registry_has_unique_names_and_alias_classification(self) -> None:
        rows = feature_registry_rows()
        feature_names = [row["feature_name"] for row in rows]
        self.assertEqual(len(feature_names), len(set(feature_names)))

        registry = {row["feature_name"]: row for row in rows}
        self.assertEqual(registry["raw_lower_face_height_px"]["status"], "deprecated_but_kept_for_compatibility")
        self.assertEqual(registry["raw_lower_face_height_px"]["alias_of"], "mouth_center_to_chin_height")
        self.assertEqual(registry["raw_jaw_height_proxy_px"]["status"], "redundant")
        self.assertEqual(registry["canthal_tilt_deg"]["status"], "new")

    def test_feature_audit_reports_alias_and_redundant_counts(self) -> None:
        audit = feature_audit_payload()
        self.assertGreater(audit["feature_count"], 20)
        self.assertIn("deprecated_but_kept_for_compatibility", audit["status_counts"])
        self.assertIn("redundant", audit["status_counts"])


if __name__ == "__main__":
    unittest.main()
