from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def _synthetic_106_dense(self) -> np.ndarray:
        dense = np.tile(np.asarray([[60.0, 80.0]], dtype=np.float64), (106, 1))
        # contour / jaw
        for offset, index in enumerate(range(1, 17)):
            dense[index] = np.asarray([10.0 + offset * 1.8, 42.0 + offset * 7.0], dtype=np.float64)
        for offset, index in enumerate(range(17, 33)):
            dense[index] = np.asarray([110.0 - offset * 1.8, 42.0 + offset * 7.0], dtype=np.float64)
        dense[0] = np.asarray([60.0, 138.0], dtype=np.float64)
        # eyes
        dense[33:43] = np.asarray([
            [34.0, 58.0], [36.0, 56.5], [38.0, 56.0], [40.0, 56.0], [42.0, 56.5],
            [44.0, 58.0], [42.0, 59.5], [40.0, 60.0], [38.0, 60.0], [36.0, 59.5],
        ], dtype=np.float64)
        dense[87:97] = np.asarray([
            [76.0, 58.0], [78.0, 56.5], [80.0, 56.0], [82.0, 56.0], [84.0, 56.5],
            [86.0, 58.0], [84.0, 59.5], [82.0, 60.0], [80.0, 60.0], [78.0, 59.5],
        ], dtype=np.float64)
        # brows
        dense[43:52] = np.asarray([[32.0 + i * 1.8, 47.0 - abs(4 - i) * 0.4] for i in range(9)], dtype=np.float64)
        dense[97:106] = np.asarray([[72.0 + i * 1.8, 47.0 - abs(4 - i) * 0.4] for i in range(9)], dtype=np.float64)
        # mouth
        dense[52] = np.asarray([40.0, 102.0], dtype=np.float64)
        dense[53] = np.asarray([60.0, 114.0], dtype=np.float64)
        dense[61] = np.asarray([80.0, 102.0], dtype=np.float64)
        dense[71] = np.asarray([60.0, 98.0], dtype=np.float64)
        # nose
        dense[77] = np.asarray([48.0, 84.0], dtype=np.float64)
        dense[78] = np.asarray([52.0, 88.0], dtype=np.float64)
        dense[79] = np.asarray([56.0, 90.0], dtype=np.float64)
        dense[80] = np.asarray([60.0, 91.0], dtype=np.float64)
        dense[83] = np.asarray([72.0, 84.0], dtype=np.float64)
        dense[84] = np.asarray([68.0, 88.0], dtype=np.float64)
        dense[85] = np.asarray([64.0, 90.0], dtype=np.float64)
        dense[86] = np.asarray([60.0, 78.0], dtype=np.float64)
        return dense

    def _synthetic_mediapipe_dense(self) -> list[list[float]]:
        dense = np.tile(np.asarray([[60.0, 80.0, 0.0]], dtype=np.float64), (478, 1))
        mapping = {
            10: [60.0, 20.0, 0.0],
            338: [82.0, 24.0, 0.0],
            297: [92.0, 30.0, 0.0],
            332: [102.0, 40.0, 0.0],
            284: [108.0, 52.0, 0.0],
            251: [110.0, 66.0, 0.0],
            389: [112.0, 82.0, 0.0],
            356: [108.0, 98.0, 0.0],
            454: [102.0, 112.0, 0.0],
            323: [94.0, 124.0, 0.0],
            361: [84.0, 134.0, 0.0],
            288: [74.0, 142.0, 0.0],
            397: [66.0, 148.0, 0.0],
            365: [70.0, 142.0, 0.0],
            379: [74.0, 136.0, 0.0],
            378: [78.0, 132.0, 0.0],
            400: [82.0, 128.0, 0.0],
            377: [86.0, 124.0, 0.0],
            152: [60.0, 150.0, 0.0],
            148: [34.0, 142.0, 0.0],
            176: [28.0, 136.0, 0.0],
            149: [22.0, 124.0, 0.0],
            150: [18.0, 112.0, 0.0],
            136: [12.0, 98.0, 0.0],
            172: [10.0, 82.0, 0.0],
            58: [12.0, 66.0, 0.0],
            132: [18.0, 52.0, 0.0],
            93: [28.0, 40.0, 0.0],
            234: [18.0, 112.0, 0.0],
            127: [38.0, 30.0, 0.0],
            162: [48.0, 24.0, 0.0],
            21: [54.0, 22.0, 0.0],
            54: [12.0, 52.0, 0.0],
            103: [18.0, 40.0, 0.0],
            67: [28.0, 30.0, 0.0],
            109: [40.0, 24.0, 0.0],
            33: [34.0, 58.0, 0.0],
            133: [46.0, 58.0, 0.0],
            159: [40.0, 54.0, 0.0],
            145: [40.0, 62.0, 0.0],
            362: [74.0, 58.0, 0.0],
            263: [86.0, 58.0, 0.0],
            386: [80.0, 54.0, 0.0],
            374: [80.0, 62.0, 0.0],
            246: [34.0, 57.0, 0.0],
            161: [36.0, 55.5, 0.0],
            160: [38.0, 54.8, 0.0],
            158: [42.0, 54.8, 0.0],
            157: [44.0, 55.5, 0.0],
            173: [46.0, 57.0, 0.0],
            7: [35.0, 59.5, 0.0],
            163: [37.0, 61.0, 0.0],
            144: [39.0, 61.5, 0.0],
            153: [41.0, 61.5, 0.0],
            154: [43.0, 61.0, 0.0],
            155: [45.0, 59.5, 0.0],
            466: [74.0, 57.0, 0.0],
            388: [76.0, 55.5, 0.0],
            387: [78.0, 54.8, 0.0],
            385: [82.0, 54.8, 0.0],
            384: [84.0, 55.5, 0.0],
            398: [86.0, 57.0, 0.0],
            382: [75.0, 59.5, 0.0],
            381: [77.0, 61.0, 0.0],
            380: [79.0, 61.5, 0.0],
            373: [81.0, 61.5, 0.0],
            390: [83.0, 61.0, 0.0],
            249: [85.0, 59.5, 0.0],
            70: [32.0, 48.0, 0.0],
            63: [35.0, 47.0, 0.0],
            105: [38.0, 46.0, 0.0],
            66: [41.0, 45.0, 0.0],
            107: [44.0, 46.0, 0.0],
            55: [46.0, 47.0, 0.0],
            65: [43.0, 46.5, 0.0],
            52: [40.0, 46.2, 0.0],
            53: [37.0, 46.5, 0.0],
            46: [34.0, 47.0, 0.0],
            336: [74.0, 48.0, 0.0],
            296: [77.0, 47.0, 0.0],
            334: [80.0, 46.0, 0.0],
            293: [83.0, 45.0, 0.0],
            300: [86.0, 46.0, 0.0],
            285: [88.0, 47.0, 0.0],
            295: [85.0, 46.5, 0.0],
            282: [82.0, 46.2, 0.0],
            283: [79.0, 46.5, 0.0],
            276: [76.0, 47.0, 0.0],
            61: [38.0, 102.0, 0.0],
            291: [82.0, 102.0, 0.0],
            13: [60.0, 98.0, 0.0],
            14: [60.0, 106.0, 0.0],
            0: [60.0, 96.0, 0.0],
            17: [60.0, 110.0, 0.0],
            82: [54.0, 98.0, 0.0],
            312: [66.0, 98.0, 0.0],
            146: [42.0, 100.0, 0.0],
            91: [48.0, 98.5, 0.0],
            181: [54.0, 97.5, 0.0],
            84: [57.0, 97.0, 0.0],
            314: [63.0, 97.0, 0.0],
            405: [69.0, 97.5, 0.0],
            321: [74.0, 98.5, 0.0],
            375: [78.0, 100.0, 0.0],
            308: [79.0, 103.0, 0.0],
            324: [74.0, 105.0, 0.0],
            318: [69.0, 106.5, 0.0],
            402: [64.0, 107.0, 0.0],
            317: [60.0, 108.0, 0.0],
            87: [56.0, 108.0, 0.0],
            178: [51.0, 107.0, 0.0],
            88: [46.0, 106.5, 0.0],
            95: [42.0, 105.0, 0.0],
            78: [41.0, 103.0, 0.0],
            415: [70.0, 101.0, 0.0],
            310: [66.0, 100.0, 0.0],
            311: [63.0, 99.0, 0.0],
            81: [57.0, 99.0, 0.0],
            80: [54.0, 100.0, 0.0],
            191: [50.0, 101.0, 0.0],
            1: [60.0, 78.0, 0.0],
            6: [60.0, 60.0, 0.0],
            129: [51.0, 82.0, 0.0],
            358: [69.0, 82.0, 0.0],
            98: [54.0, 88.0, 0.0],
            327: [66.0, 88.0, 0.0],
            193: [56.0, 66.0, 0.0],
            417: [64.0, 66.0, 0.0],
        }
        for index, value in mapping.items():
            dense[index] = np.asarray(value, dtype=np.float64)
        return dense.tolist()

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
        self.assertLess(raw["philtrum_width_proxy"], raw["raw_mouth_width_px"])

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

    def test_mediapipe_measurements_add_runtime_features(self) -> None:
        mediapipe_landmarks = self._synthetic_mediapipe_dense()
        raw = extract_raw_measurements(
            self.landmarks,
            self.bbox,
            self.landmarks_dense,
            mediapipe_landmarks=mediapipe_landmarks,
        )
        normalized = extract_normalized_measurements(raw)
        symmetry = extract_symmetry_features(raw, normalized)
        angles = extract_angle_features(
            self.landmarks,
            {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            self.landmarks_dense,
            self.bbox,
            mediapipe_landmarks=mediapipe_landmarks,
        )

        self.assertIsNotNone(raw["mediapipe_face_oval_width"])
        self.assertIsNotNone(raw["mediapipe_left_eye_width"])
        self.assertIsNotNone(raw["mediapipe_nose_width"])
        self.assertIsNotNone(raw["mediapipe_mouth_width"])
        self.assertIsNotNone(raw["mediapipe_jaw_width_proxy"])
        self.assertIsNotNone(normalized["mediapipe_face_oval_aspect_ratio"])
        self.assertIsNotNone(symmetry["mediapipe_eye_shape_symmetry_index"])
        self.assertIsNotNone(angles["mediapipe_nose_axis_angle_deg"])

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

    def test_semantic_face_extents_drive_face_width_and_height_in_106_path(self) -> None:
        dense = self._synthetic_106_dense()
        landmarks = [
            [39.0, 58.0],
            [81.0, 58.0],
            [60.0, 78.0],
            [40.0, 102.0],
            [80.0, 102.0],
        ]
        bbox = [0.0, 0.0, 120.0, 160.0]

        bundle = derive_semantic_landmark_bundle(
            landmarks=landmarks,
            landmarks_dense=dense.tolist(),
            bbox=bbox,
        )
        raw = extract_raw_measurements(landmarks, bbox, dense.tolist())

        self.assertNotAlmostEqual(raw["raw_face_width_px"], 120.0)
        self.assertNotAlmostEqual(raw["raw_face_height_px"], 160.0)
        self.assertLess(raw["raw_face_width_px"], 120.0)
        self.assertLess(raw["raw_face_height_px"], 160.0)
        self.assertEqual(bundle["semantic_points"]["nose_tip"]["source_index_106"], 86)
        self.assertEqual(bundle["semantic_points"]["nose_base_left"]["source_index_106"], 78)
        self.assertEqual(bundle["semantic_points"]["nose_base_right"]["source_index_106"], 84)
        self.assertEqual(bundle["semantic_points"]["subnasale_center"]["source_index_106"], 80)
        self.assertEqual(bundle["semantic_points"]["nose_mid_left"]["source_index_106"], 77)
        self.assertEqual(bundle["semantic_points"]["nose_mid_right"]["source_index_106"], 83)
        self.assertEqual(bundle["semantic_points"]["nose_tip"]["source_rule"], "canonical_index_86")

    def test_corrected_nose_length_uses_bridge_to_subnasale_not_adjacent_base_points(self) -> None:
        dense = self._synthetic_106_dense()
        landmarks = [
            [39.0, 58.0],
            [81.0, 58.0],
            [60.0, 78.0],
            [40.0, 102.0],
            [80.0, 102.0],
        ]
        raw = extract_raw_measurements(landmarks, [0.0, 0.0, 120.0, 160.0], dense.tolist())
        adjacent_low_nose_span = float(np.linalg.norm(dense[86] - ((dense[78] + dense[84]) / 2.0)))
        corrected_length = float(np.linalg.norm(dense[72] - dense[80]))
        self.assertGreater(raw["nose_length"], adjacent_low_nose_span)
        self.assertAlmostEqual(raw["nose_length"], corrected_length, places=4)

    def test_jaw_angle_asymmetry_is_mirror_aware(self) -> None:
        context = {
            "left_eye_center": np.asarray([30.0, 40.0], dtype=np.float64),
            "right_eye_center": np.asarray([70.0, 40.0], dtype=np.float64),
            "nose_tip": np.asarray([50.0, 60.0], dtype=np.float64),
            "left_mouth": np.asarray([36.0, 80.0], dtype=np.float64),
            "right_mouth": np.asarray([64.0, 80.0], dtype=np.float64),
            "mouth_center": np.asarray([50.0, 80.0], dtype=np.float64),
            "eye_center": np.asarray([50.0, 40.0], dtype=np.float64),
            "nose_base_center": np.asarray([50.0, 68.0], dtype=np.float64),
            "chin_center": np.asarray([50.0, 100.0], dtype=np.float64),
            "jaw_left": np.asarray([35.0, 90.0], dtype=np.float64),
            "jaw_right": np.asarray([64.0, 89.5], dtype=np.float64),
            "semantic_validation": {"passed": True, "warnings": [], "failed_checks": [], "warning_checks": [], "unreliable_features": []},
        }
        with patch("src.face_aesthetics.features.extraction._derive_dense_context", return_value=context):
            angles = extract_angle_features(
                self.landmarks,
                {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                self.landmarks_dense,
                self.bbox,
            )
        self.assertAlmostEqual(angles["jaw_line_left_angle_proxy"], 33.6900675, places=2)
        self.assertAlmostEqual(angles["jaw_line_right_angle_proxy"], -36.8698976, places=2)
        self.assertLess(angles["jaw_angle_asymmetry"], 4.0)

    def test_axis_angle_difference_treats_opposite_directions_as_same_axis(self) -> None:
        context = {
            "left_eye_center": np.asarray([40.0, 50.0], dtype=np.float64),
            "right_eye_center": np.asarray([60.0, 50.0], dtype=np.float64),
            "nose_tip": np.asarray([50.0, 70.0], dtype=np.float64),
            "left_mouth": np.asarray([40.0, 90.0], dtype=np.float64),
            "right_mouth": np.asarray([60.0, 90.0], dtype=np.float64),
            "mouth_center": np.asarray([50.0, 90.0], dtype=np.float64),
            "eye_center": np.asarray([50.0, 50.0], dtype=np.float64),
            "nose_base_center": np.asarray([50.0, 80.0], dtype=np.float64),
            "chin_center": np.asarray([50.0, 110.0], dtype=np.float64),
            "semantic_validation": {"passed": True, "warnings": [], "failed_checks": [], "warning_checks": [], "unreliable_features": []},
        }
        with patch("src.face_aesthetics.features.extraction._derive_dense_context", return_value=context):
            angles = extract_angle_features(
                self.landmarks,
                {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                self.landmarks_dense,
                self.bbox,
            )
        self.assertAlmostEqual(angles["angle_eye_line_deg"], 0.0)
        self.assertAlmostEqual(angles["angle_mouth_line_deg"], 0.0)
        self.assertAlmostEqual(angles["eye_line_vs_mouth_line_diff"], 0.0)

    def test_axis_angle_differences_use_orientation_not_direction(self) -> None:
        dense = self._synthetic_106_dense()
        landmarks = [
            [39.0, 58.0],
            [81.0, 58.0],
            [60.0, 78.0],
            [40.0, 102.0],
            [80.0, 102.0],
        ]
        angles = extract_angle_features(
            landmarks,
            {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            dense.tolist(),
            [0.0, 0.0, 120.0, 160.0],
        )
        self.assertAlmostEqual(angles["face_centerline_vs_nose_angle"], 0.0, places=4)
        self.assertAlmostEqual(angles["nose_to_chin_axis_diff"], 0.0, places=4)

    def test_jaw_warning_does_not_drop_valid_mouth_features(self) -> None:
        context = {
            "left_eye_center": np.asarray([30.0, 40.0], dtype=np.float64),
            "right_eye_center": np.asarray([70.0, 40.0], dtype=np.float64),
            "nose_tip": np.asarray([50.0, 60.0], dtype=np.float64),
            "left_mouth": np.asarray([36.0, 80.0], dtype=np.float64),
            "right_mouth": np.asarray([64.0, 80.0], dtype=np.float64),
            "mouth_center": np.asarray([50.0, 80.0], dtype=np.float64),
            "eye_center": np.asarray([50.0, 40.0], dtype=np.float64),
            "semantic_face_extents": {
                "left": 20.0,
                "right": 80.0,
                "top": 20.0,
                "bottom": 100.0,
                "width": 60.0,
                "height": 80.0,
            },
            "display_face_box": [20.0, 20.0, 80.0, 100.0],
            "face_center_x": 50.0,
            "brow_center": np.asarray([50.0, 30.0], dtype=np.float64),
            "nose_base_center": np.asarray([50.0, 68.0], dtype=np.float64),
            "upper_lip_top": np.asarray([50.0, 76.0], dtype=np.float64),
            "lower_lip_bottom": np.asarray([50.0, 84.0], dtype=np.float64),
            "chin_center": np.asarray([50.0, 96.0], dtype=np.float64),
            "jaw_left": np.asarray([38.0, 79.0], dtype=np.float64),
            "jaw_right": np.asarray([62.0, 84.0], dtype=np.float64),
            "semantic_validation": {
                "passed": True,
                "warnings": ["semantic_warning:jaw_left_not_below_mouth"],
                "failed_checks": [],
                "warning_checks": ["jaw_left_not_below_mouth"],
                "unreliable_features": ["raw_jaw_height_proxy_px", "jaw_line_left_angle_proxy"],
            },
        }

        with patch("src.face_aesthetics.features.extraction._derive_dense_context", return_value=context):
            raw = extract_raw_measurements(self.landmarks, self.bbox, self.landmarks_dense)
            angles = extract_angle_features(
                self.landmarks,
                {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                self.landmarks_dense,
                self.bbox,
            )

        self.assertAlmostEqual(raw["raw_mouth_width_px"], 28.0)
        self.assertAlmostEqual(raw["raw_nose_to_mouth_px"], 20.0)
        self.assertIsNone(raw["raw_jaw_height_proxy_px"])
        self.assertAlmostEqual(angles["angle_mouth_line_deg"], 0.0)
        self.assertAlmostEqual(angles["angle_eye_center_to_mouth_center_deg"], 90.0)
        self.assertIsNone(angles["jaw_line_left_angle_proxy"])

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
