from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from src.face_aesthetics.landmarks.insightface_106 import (
    ALARE_LEFT_INDEX,
    ALARE_RIGHT_INDEX,
    CONTOUR_INDICES,
    LANDMARK_COUNT as INSIGHTFACE_106_LANDMARK_COUNT,
    LEFT_BROW_INDICES,
    LEFT_CONTOUR_INDICES,
    LEFT_EYE_INDICES,
    LOWER_LIP_ARC_INDICES,
    LOWER_LIP_CENTER_BOTTOM_INDEX,
    MOUTH_INDICES,
    MOUTH_LEFT_CORNER_INDEX,
    MOUTH_RIGHT_CORNER_INDEX,
    MOUTH_INNER_ZONE_INDICES,
    NOSE_BASE_LEFT_INDEX,
    NOSE_BASE_RIGHT_INDEX,
    NOSE_BRIDGE_INDICES,
    NOSE_INDICES,
    NOSE_TIP_INDEX,
    RIGHT_BROW_INDICES,
    RIGHT_CONTOUR_INDICES,
    RIGHT_EYE_INDICES,
    SUBNASALE_CENTER_INDEX,
    SUBNASALE_LEFT_INDEX,
    SUBNASALE_RIGHT_INDEX,
    UPPER_LIP_ARC_INDICES,
    UPPER_LIP_CENTER_TOP_INDEX,
    has_106_landmarks,
    point_for_index,
    points_for_indices,
    points_for_region,
    region_name_for_index,
)


RAW_MEASUREMENT_COLUMNS = [
    "raw_face_width_px",
    "raw_face_height_px",
    "raw_eye_distance_px",
    "raw_mouth_width_px",
    "raw_eye_to_nose_px",
    "raw_nose_to_mouth_px",
    "raw_eye_to_mouth_px",
    "raw_left_eye_to_nose_px",
    "raw_right_eye_to_nose_px",
    "raw_left_mouth_to_nose_px",
    "raw_right_mouth_to_nose_px",
    "raw_upper_face_height_px",
    "raw_lower_face_height_px",
    "raw_forehead_height_proxy_px",
    "raw_jaw_height_proxy_px",
    "raw_nose_center_offset_px",
    "raw_mouth_center_offset_px",
    "brow_to_eye_center_height",
    "eye_center_to_nose_base_height",
    "nose_base_to_mouth_center_height",
    "mouth_center_to_chin_height",
    "brow_to_nose_base_height",
    "nose_base_to_chin_height",
    "eye_center_to_chin_height",
    "forehead_proxy_to_brow_height",
    "upper_third_proxy",
    "middle_third_proxy",
    "lower_third_proxy",
    "intercanthal_distance",
    "outercanthal_distance",
    "left_eye_width",
    "right_eye_width",
    "mean_eye_width",
    "left_eye_height",
    "right_eye_height",
    "mean_eye_height",
    "nose_width",
    "nose_base_width",
    "philtrum_width_proxy",
    "chin_width_proxy",
    "jaw_width_proxy",
    "cheekbone_width_proxy",
    "left_right_eye_y_diff",
    "left_right_eye_width_diff",
    "left_right_eye_height_diff",
    "left_right_brow_height_diff",
    "left_right_mouth_corner_y_diff",
    "left_right_mouth_to_nose_diff",
    "left_right_eye_to_nose_diff",
    "nose_tip_midline_offset",
    "mouth_center_midline_offset",
    "chin_center_midline_offset",
    "philtrum_midline_offset_proxy",
    "left_brow_to_eye_distance",
    "right_brow_to_eye_distance",
    "mean_brow_to_eye_distance",
    "eye_area_proxy_left",
    "eye_area_proxy_right",
    "eye_area_proxy_mean",
    "nose_length",
    "nose_tip_to_upper_lip_distance",
    "philtrum_height",
    "upper_lip_height",
    "lower_lip_height",
    "chin_height",
    "mediapipe_face_oval_width",
    "mediapipe_face_oval_height",
    "mediapipe_left_eye_width",
    "mediapipe_right_eye_width",
    "mediapipe_left_eye_height",
    "mediapipe_right_eye_height",
    "mediapipe_left_eye_area_proxy",
    "mediapipe_right_eye_area_proxy",
    "mediapipe_left_upper_eyelid_curve_proxy",
    "mediapipe_right_upper_eyelid_curve_proxy",
    "mediapipe_left_lower_eyelid_curve_proxy",
    "mediapipe_right_lower_eyelid_curve_proxy",
    "mediapipe_left_brow_width",
    "mediapipe_right_brow_width",
    "mediapipe_left_brow_arch_height_proxy",
    "mediapipe_right_brow_arch_height_proxy",
    "mediapipe_nose_width",
    "mediapipe_nose_base_width",
    "mediapipe_nose_length",
    "mediapipe_nose_bridge_width_proxy",
    "mediapipe_nose_midline_offset",
    "mediapipe_mouth_width",
    "mediapipe_mouth_height",
    "mediapipe_upper_lip_height",
    "mediapipe_lower_lip_height",
    "mediapipe_total_lip_height",
    "mediapipe_upper_lip_area_proxy",
    "mediapipe_lower_lip_area_proxy",
    "mediapipe_cupid_bow_width_proxy",
    "mediapipe_cupid_bow_depth_proxy",
    "mediapipe_cheekbone_width_proxy",
    "mediapipe_midface_width_proxy",
    "mediapipe_midface_height_proxy",
    "mediapipe_jaw_width_proxy",
    "mediapipe_chin_width_proxy",
    "mediapipe_chin_height_proxy",
    "mediapipe_lower_face_contour_length_proxy",
    "mediapipe_left_face_contour_length_proxy",
    "mediapipe_right_face_contour_length_proxy",
    "mediapipe_chin_midline_offset",
]

SEMANTIC_UNRELIABLE_RAW_FEATURES = {
    "raw_mouth_width_px",
    "raw_nose_to_mouth_px",
    "raw_eye_to_mouth_px",
    "raw_left_mouth_to_nose_px",
    "raw_right_mouth_to_nose_px",
    "raw_lower_face_height_px",
    "raw_jaw_height_proxy_px",
    "raw_mouth_center_offset_px",
    "nose_base_to_mouth_center_height",
    "mouth_center_to_chin_height",
    "left_right_mouth_corner_y_diff",
    "left_right_mouth_to_nose_diff",
    "mouth_center_midline_offset",
    "philtrum_midline_offset_proxy",
    "nose_tip_to_upper_lip_distance",
    "philtrum_height",
    "upper_lip_height",
    "lower_lip_height",
    "chin_height",
    "chin_width_proxy",
    "jaw_width_proxy",
    "cheekbone_width_proxy",
}

SEMANTIC_UNRELIABLE_ANGLE_FEATURES = {
    "angle_mouth_line_deg",
    "angle_eye_center_to_mouth_center_deg",
    "angle_nose_to_mouth_center_deg",
    "nose_axis_angle_deg",
    "philtrum_axis_angle_deg",
    "chin_axis_angle_deg",
    "jaw_line_left_angle_proxy",
    "jaw_line_right_angle_proxy",
    "jaw_angle_asymmetry",
    "face_centerline_vs_nose_angle",
    "eye_line_vs_mouth_line_diff",
    "nose_to_chin_axis_diff",
    "lower_face_tilt_proxy",
}

SEMANTIC_WARNING_CHECKS = {
    "jaw_left_not_below_mouth",
    "jaw_right_not_below_mouth",
}

SEMANTIC_WARNING_UNRELIABLE_FEATURES = {
}

NORMALIZED_MEASUREMENT_COLUMNS = [
    "ratio_face_aspect",
    "ratio_eye_distance_to_face_width",
    "ratio_mouth_width_to_face_width",
    "ratio_eye_to_nose_to_face_height",
    "ratio_nose_to_mouth_to_face_height",
    "ratio_eye_to_mouth_to_face_height",
    "ratio_eye_distance_to_mouth_width",
    "ratio_upper_face_height_to_face_height",
    "ratio_lower_face_height_to_face_height",
    "ratio_forehead_height_proxy_to_face_height",
    "ratio_jaw_height_proxy_to_face_height",
    "ratio_nose_center_offset_to_face_width",
    "ratio_mouth_center_offset_to_face_width",
    "ratio_upper_third_to_middle_third",
    "ratio_middle_third_to_lower_third",
    "ratio_nose_base_to_mouth_to_lower_third",
    "ratio_mouth_to_chin_to_lower_third",
    "thirds_balance_index",
    "ratio_intercanthal_to_eye_width",
    "ratio_outercanthal_to_face_width",
    "ratio_nose_width_to_face_width",
    "ratio_mouth_width_to_nose_width",
    "ratio_chin_width_to_jaw_width",
    "ratio_cheekbone_width_to_jaw_width",
    "ratio_eye_height_to_eye_width",
    "ratio_eye_spacing_to_eye_width",
    "ratio_eye_spacing_to_face_width",
    "ratio_nose_length_to_face_height",
    "ratio_philtrum_to_lower_face_height",
    "ratio_upper_lip_to_lower_lip",
    "ratio_mouth_to_chin",
    "ratio_nose_to_philtrum",
    "ratio_nose_width_to_mouth_width",
    "lip_thickness_ratio",
    "chin_taper_ratio",
    "mediapipe_face_oval_aspect_ratio",
    "mediapipe_left_eye_width_to_face_width",
    "mediapipe_right_eye_width_to_face_width",
    "mediapipe_mean_eye_height_to_width",
    "mediapipe_nose_width_to_face_width",
    "mediapipe_nose_length_to_face_height",
    "mediapipe_mouth_width_to_face_width",
    "mediapipe_total_lip_height_to_mouth_width",
    "mediapipe_cheekbone_width_to_jaw_ratio",
    "mediapipe_midface_width_to_face_width",
    "mediapipe_midface_height_to_face_height",
    "mediapipe_jaw_width_to_face_width",
    "mediapipe_chin_width_to_jaw_width",
    "mediapipe_jaw_taper_ratio",
    "mediapipe_chin_taper_ratio",
    "mediapipe_lip_area_ratio",
    "mediapipe_face_height_to_width",
    "mediapipe_face_width_to_height",
    "mediapipe_lower_face_height_to_face_height",
    "mediapipe_face_contour_left_right_length_ratio",
    "mediapipe_face_contour_asymmetry_ratio",
    "mediapipe_left_eye_aspect_ratio",
    "mediapipe_right_eye_aspect_ratio",
    "mediapipe_mean_eye_aspect_ratio",
    "mediapipe_left_eye_height_to_face_height",
    "mediapipe_right_eye_height_to_face_height",
    "mediapipe_mean_eye_height_to_face_height",
    "mediapipe_intercanthal_to_face_width",
    "mediapipe_outercanthal_to_face_width",
    "mediapipe_intercanthal_to_eye_width",
    "mediapipe_eye_spacing_balance_index",
    "mediapipe_left_brow_width_to_face_width",
    "mediapipe_right_brow_width_to_face_width",
    "mediapipe_mean_brow_width_to_face_width",
    "mediapipe_brow_width_symmetry_ratio",
    "mediapipe_brow_arch_symmetry_ratio",
    "mediapipe_nose_base_width_to_face_width",
    "mediapipe_nose_bridge_width_to_face_width",
    "mediapipe_nose_width_to_mouth_width",
    "mediapipe_mouth_width_to_nose_width",
    "mediapipe_nose_width_to_intercanthal",
    "mediapipe_nose_base_width_to_intercanthal",
    "mediapipe_nose_midline_offset_to_face_width",
    "mediapipe_mouth_width_to_jaw_width",
    "mediapipe_mouth_width_to_cheekbone_width",
    "mediapipe_mouth_width_to_intercanthal",
    "mediapipe_mouth_opening_to_mouth_width",
    "mediapipe_upper_lip_height_to_mouth_width",
    "mediapipe_lower_lip_height_to_mouth_width",
    "mediapipe_upper_lip_to_lower_lip_height_ratio",
    "mediapipe_lip_fullness_index",
    "mediapipe_cupid_bow_width_to_mouth_width",
    "mediapipe_cupid_bow_depth_to_upper_lip_height",
    "mediapipe_cheekbone_width_to_face_width",
    "mediapipe_cheekbone_width_to_mouth_width",
    "mediapipe_cheekbone_width_to_nose_width",
    "mediapipe_midface_width_to_jaw_width",
    "mediapipe_jaw_width_to_cheekbone_width",
    "mediapipe_chin_width_to_face_width",
    "mediapipe_chin_height_to_face_height",
    "mediapipe_chin_midline_offset_to_face_width",
    "mediapipe_lower_face_width_to_midface_width",
    "mediapipe_upper_to_middle_third_ratio",
    "mediapipe_middle_to_lower_third_ratio",
    "mediapipe_upper_to_lower_third_ratio",
    "mediapipe_thirds_balance_index",
    "mediapipe_nose_to_mouth_to_face_height",
    "mediapipe_eye_to_mouth_to_face_height",
    "mediapipe_mouth_to_chin_to_lower_face",
    "mediapipe_face_oval_area_proxy",
    "mediapipe_face_oval_compactness_proxy",
    "mediapipe_face_taper_index",
    "mediapipe_lower_face_taper_index",
    "mediapipe_upper_face_height_to_face_height",
    "mediapipe_forehead_proxy_to_face_height",
    "mediapipe_mean_eye_width_to_face_width",
    "mediapipe_eye_outer_distance_to_face_width",
    "mediapipe_left_eye_area_to_face_area_proxy",
    "mediapipe_right_eye_area_to_face_area_proxy",
    "mediapipe_eye_area_symmetry_ratio",
    "mediapipe_eye_width_symmetry_ratio",
    "mediapipe_eye_height_symmetry_ratio",
    "mediapipe_left_brow_to_eye_distance",
    "mediapipe_right_brow_to_eye_distance",
    "mediapipe_brow_to_eye_distance_symmetry",
    "mediapipe_brow_to_eye_distance_to_eye_height",
    "mediapipe_left_brow_arch_to_eye_height",
    "mediapipe_right_brow_arch_to_eye_height",
    "mediapipe_brow_inner_distance_to_face_width",
    "mediapipe_brow_outer_distance_to_face_width",
    "mediapipe_nose_length_to_midface_height",
    "mediapipe_nose_length_to_lower_face_height",
    "mediapipe_nose_symmetry_to_face_width",
    "mediapipe_nose_tip_to_upper_lip_to_lower_face",
    "mediapipe_upper_lip_area_to_lower_lip_area_ratio",
    "mediapipe_lip_area_to_face_area_proxy",
    "mediapipe_mouth_corner_y_diff_to_mouth_width",
    "mediapipe_cheekbone_width_to_jaw_width",
    "mediapipe_cheekbone_height_to_face_height",
    "mediapipe_midface_to_lower_face_height_ratio",
    "mediapipe_cheekbone_prominence_proxy",
    "mediapipe_chin_height_to_lower_face_height",
    "mediapipe_jaw_squareness_proxy",
    "mediapipe_lower_face_width_to_face_width",
    "mediapipe_jaw_left_right_length_ratio",
    "mediapipe_horizontal_fifths_balance_proxy",
    "mediapipe_eye_to_nose_to_face_height",
    "mediapipe_nose_base_to_mouth_to_lower_face",
]

SYMMETRY_FEATURE_COLUMNS = [
    "sym_eye_nose_delta_ratio",
    "sym_mouth_nose_delta_ratio",
    "sym_horizontal_center_offset_ratio",
    "symmetry_index",
    "brow_symmetry_ratio",
    "midline_alignment_index",
    "upper_face_symmetry_index",
    "lower_face_symmetry_index",
    "mediapipe_eye_shape_symmetry_index",
    "mediapipe_brow_symmetry_index",
    "mediapipe_nose_symmetry_index",
    "mediapipe_lip_symmetry_index",
    "mediapipe_face_contour_symmetry_index",
    "mediapipe_eye_symmetry_composite",
    "mediapipe_brow_symmetry_composite",
    "mediapipe_nose_symmetry_composite",
    "mediapipe_lip_symmetry_composite",
    "mediapipe_jaw_chin_symmetry_composite",
    "mediapipe_face_contour_symmetry_composite",
    "mediapipe_global_symmetry_index",
]

ANGLE_FEATURE_COLUMNS = [
    "angle_eye_line_deg",
    "angle_mouth_line_deg",
    "angle_eye_center_to_mouth_center_deg",
    "angle_nose_to_mouth_center_deg",
    "pose_yaw_deg",
    "pose_pitch_deg",
    "pose_roll_deg",
    "canthal_tilt_deg",
    "nose_axis_angle_deg",
    "philtrum_axis_angle_deg",
    "chin_axis_angle_deg",
    "brow_line_angle_deg",
    "jaw_line_left_angle_proxy",
    "jaw_line_right_angle_proxy",
    "jaw_angle_asymmetry",
    "face_centerline_vs_nose_angle",
    "eye_line_vs_mouth_line_diff",
    "nose_to_chin_axis_diff",
    "lower_face_tilt_proxy",
    "mediapipe_left_brow_angle_deg",
    "mediapipe_right_brow_angle_deg",
    "mediapipe_nose_axis_angle_deg",
    "mediapipe_nose_axis_vs_face_midline_deg",
    "mediapipe_mouth_line_angle_deg",
    "mediapipe_canthal_tilt_left_deg",
    "mediapipe_canthal_tilt_right_deg",
    "mediapipe_eye_line_angle_deg",
    "mediapipe_brow_angle_diff_abs",
    "mediapipe_mouth_line_angle_abs",
]


NONNEGATIVE_FEATURE_COLUMNS = set(RAW_MEASUREMENT_COLUMNS + NORMALIZED_MEASUREMENT_COLUMNS + SYMMETRY_FEATURE_COLUMNS)
UNIT_INTERVAL_COLUMNS = {
    "symmetry_index",
    "brow_symmetry_ratio",
    "midline_alignment_index",
    "upper_face_symmetry_index",
    "lower_face_symmetry_index",
    "thirds_balance_index",
}

FEATURE_FORMULAS: dict[str, str] = {
    "raw_face_width_px": "semantic_face_extents.right - semantic_face_extents.left",
    "raw_face_height_px": "semantic_face_extents.bottom - semantic_face_extents.top",
    "raw_eye_distance_px": "euclidean_distance(left_eye, right_eye)",
    "raw_mouth_width_px": "euclidean_distance(left_mouth, right_mouth)",
    "raw_eye_to_nose_px": "euclidean_distance(eye_center, nose_tip)",
    "raw_nose_to_mouth_px": "euclidean_distance(nose_tip, mouth_center)",
    "raw_eye_to_mouth_px": "euclidean_distance(eye_center, mouth_center)",
    "brow_to_eye_center_height": "abs(eye_center_y - brow_center_y)",
    "eye_center_to_nose_base_height": "abs(nose_base_center_y - eye_center_y)",
    "nose_base_to_mouth_center_height": "abs(mouth_center_y - nose_base_center_y)",
    "mouth_center_to_chin_height": "abs(chin_center_y - mouth_center_y)",
    "brow_to_nose_base_height": "abs(nose_base_center_y - brow_center_y)",
    "nose_base_to_chin_height": "abs(chin_center_y - nose_base_center_y)",
    "eye_center_to_chin_height": "abs(chin_center_y - eye_center_y)",
    "forehead_proxy_to_brow_height": "abs(brow_center_y - semantic_face_extents.top)",
    "upper_third_proxy": "brow_to_nose_base_height",
    "middle_third_proxy": "eye_center_to_nose_base_height + abs(brow_center_y - eye_center_y)",
    "lower_third_proxy": "nose_base_to_chin_height",
    "intercanthal_distance": "distance(left_eye_inner_corner, right_eye_inner_corner)",
    "outercanthal_distance": "distance(left_eye_outer_corner, right_eye_outer_corner)",
    "left_eye_width": "distance(left_eye_outer_corner, left_eye_inner_corner)",
    "right_eye_width": "distance(right_eye_inner_corner, right_eye_outer_corner)",
    "left_eye_height": "distance(left_eye_top, left_eye_bottom)",
    "right_eye_height": "distance(right_eye_top, right_eye_bottom)",
    "nose_width": "distance(nose_mid_left, nose_mid_right)",
    "nose_base_width": "distance(nose_base_left, nose_base_right)",
    "philtrum_width_proxy": "distance(upper_lip_left_shoulder, upper_lip_right_shoulder) using canonical InsightFace-106 upper-arc landmarks when available; otherwise unavailable",
    "chin_width_proxy": "x_span(chin_band_points)",
    "jaw_width_proxy": "distance(jaw_left, jaw_right)",
    "cheekbone_width_proxy": "distance(cheekbone_left, cheekbone_right)",
    "nose_tip_midline_offset": "abs(nose_tip_x - face_midline_x)",
    "mouth_center_midline_offset": "abs(mouth_center_x - face_midline_x)",
    "chin_center_midline_offset": "abs(chin_center_x - face_midline_x)",
    "philtrum_midline_offset_proxy": "abs(philtrum_top_x - face_midline_x)",
    "left_brow_to_eye_distance": "abs(left_eye_center_y - left_brow_y)",
    "right_brow_to_eye_distance": "abs(right_eye_center_y - right_brow_y)",
    "eye_area_proxy_left": "left_eye_width * left_eye_height",
    "eye_area_proxy_right": "right_eye_width * right_eye_height",
    "nose_length": "distance(nose_bridge_top, subnasale_center)",
    "nose_tip_to_upper_lip_distance": "distance(nose_tip, upper_lip_top)",
    "philtrum_height": "abs(upper_lip_top_y - subnasale_center_y)",
    "upper_lip_height": "abs(lip_midline_center_y - upper_lip_top_y)",
    "lower_lip_height": "abs(lower_lip_bottom_y - lip_midline_center_y)",
    "chin_height": "abs(chin_center_y - lower_lip_bottom_y)",
    "ratio_face_aspect": "raw_face_width_px / raw_face_height_px",
    "ratio_eye_distance_to_face_width": "raw_eye_distance_px / raw_face_width_px",
    "ratio_mouth_width_to_face_width": "raw_mouth_width_px / raw_face_width_px",
    "ratio_upper_third_to_middle_third": "upper_third_proxy / middle_third_proxy",
    "ratio_middle_third_to_lower_third": "middle_third_proxy / lower_third_proxy",
    "ratio_nose_base_to_mouth_to_lower_third": "nose_base_to_mouth_center_height / lower_third_proxy",
    "ratio_mouth_to_chin_to_lower_third": "mouth_center_to_chin_height / lower_third_proxy",
    "thirds_balance_index": "1 - normalized dispersion of upper/middle/lower thirds",
    "ratio_intercanthal_to_eye_width": "intercanthal_distance / mean_eye_width",
    "ratio_outercanthal_to_face_width": "outercanthal_distance / raw_face_width_px",
    "ratio_nose_width_to_face_width": "nose_width / raw_face_width_px",
    "ratio_mouth_width_to_nose_width": "raw_mouth_width_px / nose_width",
    "ratio_chin_width_to_jaw_width": "chin_width_proxy / jaw_width_proxy",
    "ratio_cheekbone_width_to_jaw_width": "cheekbone_width_proxy / jaw_width_proxy",
    "ratio_eye_height_to_eye_width": "mean_eye_height / mean_eye_width",
    "ratio_eye_spacing_to_eye_width": "intercanthal_distance / mean_eye_width",
    "ratio_eye_spacing_to_face_width": "intercanthal_distance / raw_face_width_px",
    "ratio_eye_distance_to_mouth_width": "raw_eye_distance_px / raw_mouth_width_px",
    "ratio_nose_length_to_face_height": "nose_length / raw_face_height_px",
    "ratio_philtrum_to_lower_face_height": "philtrum_height / lower_third_proxy",
    "ratio_upper_lip_to_lower_lip": "upper_lip_height / lower_lip_height",
    "ratio_mouth_to_chin": "raw_mouth_width_px / mouth_center_to_chin_height",
    "ratio_nose_to_philtrum": "nose_length / philtrum_height",
    "ratio_nose_width_to_mouth_width": "nose_width / raw_mouth_width_px",
    "lip_thickness_ratio": "(upper_lip_height + lower_lip_height) / raw_mouth_width_px",
    "chin_taper_ratio": "chin_width_proxy / jaw_width_proxy",
    "sym_eye_nose_delta_ratio": "abs(raw_left_eye_to_nose_px - raw_right_eye_to_nose_px) / raw_eye_distance_px",
    "sym_mouth_nose_delta_ratio": "abs(raw_left_mouth_to_nose_px - raw_right_mouth_to_nose_px) / raw_mouth_width_px",
    "sym_horizontal_center_offset_ratio": "mean(nose_tip_midline_offset, mouth_center_midline_offset) / raw_face_width_px",
    "symmetry_index": "mean(1 - clipped asymmetry components)",
    "brow_symmetry_ratio": "1 - abs(left_brow_to_eye_distance - right_brow_to_eye_distance) / mean_brow_to_eye_distance",
    "midline_alignment_index": "1 - mean(midline offsets normalized by face width)",
    "upper_face_symmetry_index": "mean(1 - normalized upper-face asymmetry components)",
    "lower_face_symmetry_index": "mean(1 - normalized lower-face asymmetry components)",
    "angle_eye_line_deg": "atan2(dy, dx) between left_eye and right_eye",
    "angle_mouth_line_deg": "atan2(dy, dx) between left_mouth and right_mouth",
    "angle_eye_center_to_mouth_center_deg": "atan2(dy, dx) between eye_center and mouth_center",
    "angle_nose_to_mouth_center_deg": "atan2(dy, dx) between nose_tip and mouth_center",
    "canthal_tilt_deg": "angle_eye_line_deg",
    "nose_axis_angle_deg": "orientation-normalized atan2(dy, dx) between nose_base_center and nose_tip",
    "philtrum_axis_angle_deg": "orientation-normalized atan2(dy, dx) between nose_base_center and mouth_center",
    "chin_axis_angle_deg": "orientation-normalized atan2(dy, dx) between mouth_center and chin_center",
    "brow_line_angle_deg": "atan2(dy, dx) between left_brow and right_brow",
    "jaw_line_left_angle_proxy": "atan2(dy, dx) between jaw_left and chin_center",
    "jaw_line_right_angle_proxy": "atan2(dy, dx) between chin_center and jaw_right",
    "jaw_angle_asymmetry": "abs(abs(jaw_line_left_angle_proxy) - abs(jaw_line_right_angle_proxy))",
    "face_centerline_vs_nose_angle": "minimal orientation difference between vertical face midline and nose axis",
    "eye_line_vs_mouth_line_diff": "abs(angle_eye_line_deg - angle_mouth_line_deg)",
    "nose_to_chin_axis_diff": "minimal orientation difference between nose axis and chin axis",
    "lower_face_tilt_proxy": "chin_axis_angle_deg",
    "mediapipe_face_oval_width": "distance(x_extrema(mediapipe_face_oval))",
    "mediapipe_face_oval_height": "distance(y_extrema(mediapipe_face_oval))",
    "mediapipe_left_eye_width": "distance(mediapipe_left_eye_outer, mediapipe_left_eye_inner)",
    "mediapipe_right_eye_width": "distance(mediapipe_right_eye_inner, mediapipe_right_eye_outer)",
    "mediapipe_left_eye_height": "distance(mediapipe_left_eye_top, mediapipe_left_eye_bottom)",
    "mediapipe_right_eye_height": "distance(mediapipe_right_eye_top, mediapipe_right_eye_bottom)",
    "mediapipe_left_eye_area_proxy": "mediapipe_left_eye_width * mediapipe_left_eye_height",
    "mediapipe_right_eye_area_proxy": "mediapipe_right_eye_width * mediapipe_right_eye_height",
    "mediapipe_left_upper_eyelid_curve_proxy": "polyline_length(mediapipe_left_upper_lid) / mediapipe_left_eye_width",
    "mediapipe_right_upper_eyelid_curve_proxy": "polyline_length(mediapipe_right_upper_lid) / mediapipe_right_eye_width",
    "mediapipe_left_lower_eyelid_curve_proxy": "polyline_length(mediapipe_left_lower_lid) / mediapipe_left_eye_width",
    "mediapipe_right_lower_eyelid_curve_proxy": "polyline_length(mediapipe_right_lower_lid) / mediapipe_right_eye_width",
    "mediapipe_left_brow_width": "x_span(mediapipe_left_brow_arc)",
    "mediapipe_right_brow_width": "x_span(mediapipe_right_brow_arc)",
    "mediapipe_left_brow_arch_height_proxy": "arch_height(mediapipe_left_brow_arc)",
    "mediapipe_right_brow_arch_height_proxy": "arch_height(mediapipe_right_brow_arc)",
    "mediapipe_nose_width": "distance(mediapipe_nose_left, mediapipe_nose_right)",
    "mediapipe_nose_base_width": "distance(mediapipe_nose_base_left, mediapipe_nose_base_right)",
    "mediapipe_nose_length": "distance(mediapipe_nose_bridge_top, mediapipe_nose_tip)",
    "mediapipe_nose_bridge_width_proxy": "distance(bridged nasal side proxies)",
    "mediapipe_nose_midline_offset": "abs(mediapipe_nose_tip_x - mediapipe_face_oval_center_x)",
    "mediapipe_mouth_width": "distance(mediapipe_mouth_left, mediapipe_mouth_right)",
    "mediapipe_mouth_height": "distance(inner_upper_lip_center, inner_lower_lip_center) (mouth opening height proxy)",
    "mediapipe_upper_lip_height": "distance(upper_outer_lip_center, upper_inner_lip_center)",
    "mediapipe_lower_lip_height": "distance(lower_inner_lip_center, lower_outer_lip_center)",
    "mediapipe_total_lip_height": "mediapipe_upper_lip_height + mediapipe_lower_lip_height",
    "mediapipe_upper_lip_area_proxy": "polygon_area(upper lip outer contour) with width*height fallback",
    "mediapipe_lower_lip_area_proxy": "polygon_area(lower lip outer contour) with width*height fallback",
    "mediapipe_cupid_bow_width_proxy": "distance(mediapipe_cupid_left, mediapipe_cupid_right)",
    "mediapipe_cupid_bow_depth_proxy": "distance(mediapipe_upper_lip_top, mediapipe_lip_center)",
    "mediapipe_cheekbone_width_proxy": "distance(mediapipe_left_cheek, mediapipe_right_cheek)",
    "mediapipe_midface_width_proxy": "distance(mediapipe_left_cheek, mediapipe_right_cheek)",
    "mediapipe_midface_height_proxy": "distance(mediapipe_left_brow_center, mediapipe_lip_center)",
    "mediapipe_jaw_width_proxy": "distance(mediapipe_left_jaw, mediapipe_right_jaw)",
    "mediapipe_chin_width_proxy": "x_span(lower central face-oval band)",
    "mediapipe_chin_height_proxy": "distance(mediapipe_lip_center, mediapipe_chin)",
    "mediapipe_lower_face_contour_length_proxy": "polyline_length(lower mediapipe_face_oval arc)",
    "mediapipe_left_face_contour_length_proxy": "polyline_length(left mediapipe_face_oval arc)",
    "mediapipe_right_face_contour_length_proxy": "polyline_length(right mediapipe_face_oval arc)",
    "mediapipe_chin_midline_offset": "abs(mediapipe_chin_x - mediapipe_face_oval_center_x)",
    "mediapipe_face_oval_aspect_ratio": "mediapipe_face_oval_width / mediapipe_face_oval_height",
    "mediapipe_left_eye_width_to_face_width": "mediapipe_left_eye_width / mediapipe_face_oval_width",
    "mediapipe_right_eye_width_to_face_width": "mediapipe_right_eye_width / mediapipe_face_oval_width",
    "mediapipe_mean_eye_height_to_width": "mean(mediapipe_left_eye_height, mediapipe_right_eye_height) / mean(mediapipe_left_eye_width, mediapipe_right_eye_width)",
    "mediapipe_nose_width_to_face_width": "mediapipe_nose_width / mediapipe_face_oval_width",
    "mediapipe_nose_length_to_face_height": "mediapipe_nose_length / mediapipe_face_oval_height",
    "mediapipe_mouth_width_to_face_width": "mediapipe_mouth_width / mediapipe_face_oval_width",
    "mediapipe_total_lip_height_to_mouth_width": "mediapipe_total_lip_height / mediapipe_mouth_width",
    "mediapipe_cheekbone_width_to_jaw_ratio": "mediapipe_cheekbone_width_proxy / mediapipe_jaw_width_proxy",
    "mediapipe_midface_width_to_face_width": "mediapipe_midface_width_proxy / mediapipe_face_oval_width",
    "mediapipe_midface_height_to_face_height": "mediapipe_midface_height_proxy / mediapipe_face_oval_height",
    "mediapipe_jaw_width_to_face_width": "mediapipe_jaw_width_proxy / mediapipe_face_oval_width",
    "mediapipe_chin_width_to_jaw_width": "mediapipe_chin_width_proxy / mediapipe_jaw_width_proxy",
    "mediapipe_jaw_taper_ratio": "mediapipe_jaw_width_proxy / mediapipe_cheekbone_width_proxy",
    "mediapipe_chin_taper_ratio": "mediapipe_chin_width_proxy / mediapipe_jaw_width_proxy",
    "mediapipe_lip_area_ratio": "mediapipe_upper_lip_area_proxy / mediapipe_lower_lip_area_proxy",
    "mediapipe_eye_shape_symmetry_index": "1 - normalized difference between left and right MediaPipe eye widths/heights/areas",
    "mediapipe_brow_symmetry_index": "1 - normalized difference between left and right MediaPipe brow widths/arch heights",
    "mediapipe_nose_symmetry_index": "1 - normalized mediapipe_nose_midline_offset / mediapipe_face_oval_width",
    "mediapipe_lip_symmetry_index": "1 - normalized lip corner and height asymmetry from MediaPipe",
    "mediapipe_face_contour_symmetry_index": "1 - normalized difference between left and right MediaPipe contour lengths",
    "mediapipe_left_brow_angle_deg": "orientation-normalized atan2(dy, dx) from inner->outer left brow endpoint",
    "mediapipe_right_brow_angle_deg": "orientation-normalized atan2(dy, dx) from inner->outer right brow endpoint",
    "mediapipe_nose_axis_angle_deg": "orientation-normalized atan2(dy, dx) between mediapipe_nose_bridge_top and mediapipe_nose_tip",
    "mediapipe_nose_axis_vs_face_midline_deg": "minimal orientation difference between mediapipe_nose_axis_angle_deg and vertical face midline",
    "mediapipe_mouth_line_angle_deg": "atan2(dy, dx) between mediapipe_mouth_left and mediapipe_mouth_right",
    "mediapipe_canthal_tilt_left_deg": "atan2(dy, dx) between mediapipe_left_eye_outer and mediapipe_left_eye_inner",
    "mediapipe_canthal_tilt_right_deg": "atan2(dy, dx) between mediapipe_right_eye_inner and mediapipe_right_eye_outer",
    "mediapipe_eye_line_angle_deg": "atan2(dy, dx) between outer eye corners from MediaPipe",
    "mediapipe_brow_angle_diff_abs": "minimal axis-angle difference between left and right MediaPipe brow angles",
    "mediapipe_mouth_line_angle_abs": "absolute value of mediapipe_mouth_line_angle_deg",
    "mediapipe_face_oval_area_proxy": "mediapipe_face_oval_width * mediapipe_face_oval_height",
    "mediapipe_face_oval_compactness_proxy": "4*pi*mediapipe_face_oval_area_proxy / perimeter_proxy^2",
    "mediapipe_face_taper_index": "mediapipe_cheekbone_width_proxy / mediapipe_jaw_width_proxy",
    "mediapipe_lower_face_taper_index": "mediapipe_jaw_width_proxy / mediapipe_chin_width_proxy",
    "mediapipe_upper_face_height_to_face_height": "upper_third_proxy / mediapipe_face_oval_height",
    "mediapipe_forehead_proxy_to_face_height": "raw_forehead_height_proxy_px / mediapipe_face_oval_height",
    "mediapipe_mean_eye_width_to_face_width": "mean(mediapipe_left_eye_width, mediapipe_right_eye_width) / mediapipe_face_oval_width",
    "mediapipe_eye_outer_distance_to_face_width": "outercanthal_distance / mediapipe_face_oval_width",
    "mediapipe_left_eye_area_to_face_area_proxy": "mediapipe_left_eye_area_proxy / mediapipe_face_oval_area_proxy",
    "mediapipe_right_eye_area_to_face_area_proxy": "mediapipe_right_eye_area_proxy / mediapipe_face_oval_area_proxy",
    "mediapipe_eye_area_symmetry_ratio": "abs(mediapipe_left_eye_area_proxy - mediapipe_right_eye_area_proxy) / mean(area_left, area_right)",
    "mediapipe_eye_width_symmetry_ratio": "abs(mediapipe_left_eye_width - mediapipe_right_eye_width) / mean(width_left, width_right)",
    "mediapipe_eye_height_symmetry_ratio": "abs(mediapipe_left_eye_height - mediapipe_right_eye_height) / mean(height_left, height_right)",
    "mediapipe_left_brow_to_eye_distance": "left_brow_to_eye_distance",
    "mediapipe_right_brow_to_eye_distance": "right_brow_to_eye_distance",
    "mediapipe_brow_to_eye_distance_symmetry": "abs(left_brow_to_eye_distance - right_brow_to_eye_distance) / mean(left,right)",
    "mediapipe_brow_to_eye_distance_to_eye_height": "mean(left_brow_to_eye_distance,right_brow_to_eye_distance) / mean(mediapipe_left_eye_height,mediapipe_right_eye_height)",
    "mediapipe_left_brow_arch_to_eye_height": "mediapipe_left_brow_arch_height_proxy / mediapipe_left_eye_height",
    "mediapipe_right_brow_arch_to_eye_height": "mediapipe_right_brow_arch_height_proxy / mediapipe_right_eye_height",
    "mediapipe_brow_inner_distance_to_face_width": "intercanthal_distance / mediapipe_face_oval_width",
    "mediapipe_brow_outer_distance_to_face_width": "outercanthal_distance / mediapipe_face_oval_width",
    "mediapipe_nose_length_to_midface_height": "mediapipe_nose_length / mediapipe_midface_height_proxy",
    "mediapipe_nose_length_to_lower_face_height": "mediapipe_nose_length / lower_third_proxy",
    "mediapipe_nose_symmetry_to_face_width": "mediapipe_nose_midline_offset / mediapipe_face_oval_width",
    "mediapipe_nose_tip_to_upper_lip_to_lower_face": "mediapipe_nose_length / lower_third_proxy",
    "mediapipe_upper_lip_area_to_lower_lip_area_ratio": "mediapipe_upper_lip_area_proxy / mediapipe_lower_lip_area_proxy",
    "mediapipe_lip_area_to_face_area_proxy": "(mediapipe_upper_lip_area_proxy + mediapipe_lower_lip_area_proxy) / mediapipe_face_oval_area_proxy",
    "mediapipe_mouth_corner_y_diff_to_mouth_width": "left_right_mouth_corner_y_diff / mediapipe_mouth_width",
    "mediapipe_cheekbone_width_to_jaw_width": "mediapipe_cheekbone_width_proxy / mediapipe_jaw_width_proxy",
    "mediapipe_cheekbone_height_to_face_height": "mediapipe_midface_height_proxy / mediapipe_face_oval_height",
    "mediapipe_midface_to_lower_face_height_ratio": "mediapipe_midface_height_proxy / lower_third_proxy",
    "mediapipe_cheekbone_prominence_proxy": "mediapipe_cheekbone_width_proxy / mean(mediapipe_jaw_width_proxy, mediapipe_mouth_width)",
    "mediapipe_jaw_squareness_proxy": "mediapipe_jaw_width_proxy / mediapipe_chin_height_proxy",
    "mediapipe_lower_face_width_to_face_width": "mediapipe_jaw_width_proxy / mediapipe_face_oval_width",
    "mediapipe_jaw_left_right_length_ratio": "mediapipe_left_face_contour_length_proxy / mediapipe_right_face_contour_length_proxy",
    "mediapipe_horizontal_fifths_balance_proxy": "outercanthal_distance / mediapipe_face_oval_width",
    "mediapipe_eye_to_nose_to_face_height": "raw_eye_to_nose_px / mediapipe_face_oval_height",
    "mediapipe_nose_base_to_mouth_to_lower_face": "nose_base_to_mouth_center_height / lower_third_proxy",
}


ALIASES = {
    "raw_lower_face_height_px": "mouth_center_to_chin_height",
    "raw_jaw_height_proxy_px": "mouth_center_to_chin_height",
    "raw_nose_center_offset_px": "nose_tip_midline_offset",
    "raw_mouth_center_offset_px": "mouth_center_midline_offset",
    "angle_eye_line_deg": "canthal_tilt_deg",
}

REDUNDANT_FEATURES = {
    "raw_jaw_height_proxy_px",
}

FEATURE_METADATA_OVERRIDES: dict[str, dict[str, Any]] = {
    "nose_length": {
        "status": "corrected",
        "depends_on": ["nose_bridge_top", "subnasale_center"],
        "source_points": ["nose_bridge_top", "subnasale_center"],
        "notes": "Corrected from lower-nose-only span to a bridge-to-subnasale length proxy using canonical InsightFace-106 nose landmarks.",
        "reliability_warning": "Treat as a 2D nasal length proxy; frontal pose and landmark quality still affect absolute scale.",
    },
    "philtrum_width_proxy": {
        "status": "unstable",
        "depends_on": ["upper_lip_left_shoulder", "upper_lip_right_shoulder"],
        "source_points": ["upper_lip_left_shoulder", "upper_lip_right_shoulder"],
        "notes": "Corrected away from full mouth width, but still treated as a weak 2D proxy because InsightFace-106 does not expose a dedicated philtrum-width landmark pair.",
        "reliability_warning": "Do not use for reference comparison or strong interpretation.",
        "reference_inclusion": "excluded",
    },
    "upper_lip_height": {
        "status": "corrected",
        "depends_on": ["upper_lip_top", "lip_midline_center"],
        "source_points": ["upper_lip_top", "lip_midline_center"],
        "notes": "Corrected to use the lip midline center instead of the mouth-corner midpoint.",
    },
    "lower_lip_height": {
        "status": "corrected",
        "depends_on": ["lip_midline_center", "lower_lip_bottom"],
        "source_points": ["lip_midline_center", "lower_lip_bottom"],
        "notes": "Corrected to use the lip midline center instead of the mouth-corner midpoint.",
    },
    "jaw_angle_asymmetry": {
        "status": "corrected",
        "depends_on": ["jaw_line_left_angle_proxy", "jaw_line_right_angle_proxy"],
        "source_points": ["jaw_left", "chin_center", "jaw_right"],
        "notes": "Corrected to a mirror-aware jaw asymmetry comparison so mirrored angles do not produce false large discrepancies.",
    },
    "ratio_eye_distance_to_mouth_width": {
        "status": "corrected",
        "depends_on": ["raw_eye_distance_px", "raw_mouth_width_px"],
        "source_points": ["left_eye_center", "right_eye_center", "mouth_left", "mouth_right"],
        "notes": "Recomputed from corrected mouth corner mapping and semantic face extents pipeline.",
    },
    "ratio_mouth_width_to_face_width": {
        "status": "corrected",
        "depends_on": ["raw_mouth_width_px", "raw_face_width_px"],
        "source_points": ["mouth_left", "mouth_right", "semantic_face_extents"],
        "notes": "Recomputed from corrected mouth corner mapping and semantic face extents instead of crop size.",
    },
    "ratio_face_aspect": {
        "status": "corrected",
        "depends_on": ["raw_face_width_px", "raw_face_height_px"],
        "source_points": ["semantic_face_extents"],
        "notes": "Now derived from semantic face extents rather than normalized crop size.",
    },
}

OPTIONAL_MULTI_SOURCE_FEATURES: list[dict[str, Any]] = [
    {
        "feature_name": "full_face_oval_width",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "face_oval",
        "formula": "x_span(face_skin_mask_oval)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["skin_mask"],
        "normalized_by": None,
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires a reliable face-parsing skin mask.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "full_face_oval_height",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "face_oval",
        "formula": "y_span(face_skin_mask_oval)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["skin_mask"],
        "normalized_by": None,
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires a reliable face-parsing skin mask.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "skin_mask_face_area",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "face_oval",
        "formula": "pixel_area(face_skin_mask)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["skin_mask"],
        "normalized_by": "analysis_image_area",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires face parsing; not available from InsightFace-106 alone.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "hairline_to_brow_height",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "hairline",
        "formula": "distance(hairline_mask_midpoint, brow_center)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["hair_mask", "brow_center"],
        "normalized_by": "raw_face_height_px",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Hairline is unavailable without a face-parsing or dedicated hairline model.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "forehead_visible_height",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "hairline",
        "formula": "distance(hairline_mask_midpoint, brow_center)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["hair_mask", "brow_center"],
        "normalized_by": "raw_face_height_px",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires explicit hairline support from segmentation or a dedicated model.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "left_eye_upper_lid_curve_proxy",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "mediapipe_face_mesh",
        "component": "eyes",
        "formula": "curve_proxy(left_upper_lid_dense_arc)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["mediapipe_left_upper_lid"],
        "normalized_by": "left_eye_width",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires dense eyelid landmarks from MediaPipe Face Mesh.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "right_eye_upper_lid_curve_proxy",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "mediapipe_face_mesh",
        "component": "eyes",
        "formula": "curve_proxy(right_upper_lid_dense_arc)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["mediapipe_right_upper_lid"],
        "normalized_by": "right_eye_width",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires dense eyelid landmarks from MediaPipe Face Mesh.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "left_brow_curvature_proxy",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "mediapipe_face_mesh",
        "component": "brows",
        "formula": "curve_proxy(left_brow_dense_arc)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["mediapipe_left_brow_arc"],
        "normalized_by": "left_brow_width",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires dense brow landmarks from MediaPipe Face Mesh.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "right_brow_curvature_proxy",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "mediapipe_face_mesh",
        "component": "brows",
        "formula": "curve_proxy(right_brow_dense_arc)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["mediapipe_right_brow_arc"],
        "normalized_by": "right_brow_width",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires dense brow landmarks from MediaPipe Face Mesh.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "upper_lip_area_mask",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "mouth_lips",
        "formula": "pixel_area(upper_lip_mask)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["upper_lip_mask"],
        "normalized_by": "skin_mask_face_area",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires lip masks from face parsing.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "lower_lip_area_mask",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "mouth_lips",
        "formula": "pixel_area(lower_lip_mask)",
        "units": "index",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["lower_lip_mask"],
        "normalized_by": "skin_mask_face_area",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires lip masks from face parsing.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "visible_left_ear_height",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "ears",
        "formula": "y_span(left_ear_mask)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["left_ear_mask"],
        "normalized_by": "raw_face_height_px",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Current landmark stack does not expose ears directly; requires parsing or a dedicated ear detector.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "visible_right_ear_height",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "ears",
        "formula": "y_span(right_ear_mask)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["right_ear_mask"],
        "normalized_by": "raw_face_height_px",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Current landmark stack does not expose ears directly; requires parsing or a dedicated ear detector.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
    {
        "feature_name": "visible_ear_width",
        "feature_family": "optional_multi_source",
        "status": "unavailable_without_extra_model",
        "source": "face_parsing",
        "component": "ears",
        "formula": "x_span(visible_ear_masks)",
        "units": "px",
        "reliability": "unavailable_without_extra_model",
        "required_points": ["left_ear_mask", "right_ear_mask"],
        "normalized_by": "raw_face_width_px",
        "recommended_for_scorer": False,
        "reference_inclusion": "excluded",
        "notes": "Requires reliable ear masks; no ear landmarks are available from InsightFace-106.",
        "exclusion_reason": "unavailable_without_extra_model",
    },
]

MEDIAPIPE_FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MEDIAPIPE_LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MEDIAPIPE_RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MEDIAPIPE_LEFT_UPPER_EYELID_INDICES = [246, 161, 160, 159, 158, 157, 173]
MEDIAPIPE_RIGHT_UPPER_EYELID_INDICES = [466, 388, 387, 386, 385, 384, 398]
MEDIAPIPE_LEFT_LOWER_EYELID_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133]
MEDIAPIPE_RIGHT_LOWER_EYELID_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263]
MEDIAPIPE_LEFT_BROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MEDIAPIPE_RIGHT_BROW_INDICES = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MEDIAPIPE_LIPS_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
MEDIAPIPE_LIPS_INNER_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MEDIAPIPE_NOSE_INDICES = [6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 168]


def _mediapipe_point(points: np.ndarray | None, index: int) -> np.ndarray | None:
    if points is None or index < 0 or index >= len(points):
        return None
    point = np.asarray(points[index], dtype=np.float64)
    if point.shape[0] < 2:
        return None
    return point[:2]


def _mediapipe_points(points: np.ndarray | None, indices: list[int]) -> np.ndarray:
    if points is None:
        return np.zeros((0, 2), dtype=np.float64)
    collected: list[np.ndarray] = []
    for index in indices:
        point = _mediapipe_point(points, int(index))
        if point is not None:
            collected.append(point)
    if not collected:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(collected, dtype=np.float64)


def _polyline_length(points: np.ndarray) -> float | None:
    if len(points) < 2:
        return None
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _polygon_area(points: np.ndarray) -> float | None:
    if points is None or len(points) < 3:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area


def _brow_endpoints(points: np.ndarray | None, face_center: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(points, np.ndarray) or len(points) < 2:
        return None, None
    if face_center is None:
        return points[0], points[-1]
    distances = np.linalg.norm(points - face_center, axis=1)
    outer_idx = int(np.argmax(distances))
    inner_idx = int(np.argmin(distances))
    return points[inner_idx], points[outer_idx]


def _curve_height_proxy(points: np.ndarray, start: np.ndarray | None, end: np.ndarray | None) -> float | None:
    if len(points) == 0 or start is None or end is None:
        return None
    baseline_y = float((start[1] + end[1]) / 2.0)
    return float(abs(np.min(points[:, 1]) - baseline_y))


def _derive_mediapipe_context(
    mediapipe_landmarks: list[list[float]] | None,
) -> dict[str, Any]:
    if not mediapipe_landmarks:
        return {}
    dense = np.asarray(mediapipe_landmarks, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[1] < 2:
        return {}
    left_eye = _mediapipe_points(dense, MEDIAPIPE_LEFT_EYE_INDICES)
    right_eye = _mediapipe_points(dense, MEDIAPIPE_RIGHT_EYE_INDICES)
    left_brow = _mediapipe_points(dense, MEDIAPIPE_LEFT_BROW_INDICES)
    right_brow = _mediapipe_points(dense, MEDIAPIPE_RIGHT_BROW_INDICES)
    lips_outer = _mediapipe_points(dense, MEDIAPIPE_LIPS_OUTER_INDICES)
    lips_inner = _mediapipe_points(dense, MEDIAPIPE_LIPS_INNER_INDICES)
    face_oval = _mediapipe_points(dense, MEDIAPIPE_FACE_OVAL_INDICES)
    left_cheek = _mediapipe_point(dense, 234)
    right_cheek = _mediapipe_point(dense, 454)
    left_jaw = _mediapipe_point(dense, 172)
    right_jaw = _mediapipe_point(dense, 397)
    chin = _mediapipe_point(dense, 152)
    mouth_left = _mediapipe_point(dense, 61)
    mouth_right = _mediapipe_point(dense, 291)
    mouth_top = _mediapipe_point(dense, 13)
    mouth_bottom = _mediapipe_point(dense, 14)
    upper_lip_top = _mediapipe_point(dense, 0)
    lower_lip_bottom = _mediapipe_point(dense, 17)
    upper_lip_inner = _mediapipe_point(dense, 13)
    lower_lip_inner = _mediapipe_point(dense, 14)
    cupid_left = _mediapipe_point(dense, 82)
    cupid_right = _mediapipe_point(dense, 312)
    nose_tip = _mediapipe_point(dense, 1)
    if nose_tip is None:
        nose_tip = _mediapipe_point(dense, 4)
    nose_bridge_top = _mediapipe_point(dense, 6)
    nose_left = _mediapipe_point(dense, 129)
    nose_right = _mediapipe_point(dense, 358)
    nose_base_left = _mediapipe_point(dense, 98)
    nose_base_right = _mediapipe_point(dense, 327)
    left_eye_outer = _mediapipe_point(dense, 33)
    left_eye_inner = _mediapipe_point(dense, 133)
    right_eye_inner = _mediapipe_point(dense, 362)
    right_eye_outer = _mediapipe_point(dense, 263)
    left_eye_top = _mediapipe_point(dense, 159)
    left_eye_bottom = _mediapipe_point(dense, 145)
    right_eye_top = _mediapipe_point(dense, 386)
    right_eye_bottom = _mediapipe_point(dense, 374)
    left_brow_center = _cluster_center(left_brow)
    right_brow_center = _cluster_center(right_brow)
    mouth_center = _point_midpoint(mouth_left, mouth_right)
    lip_center = _point_midpoint(upper_lip_top, lower_lip_bottom)
    face_center = _cluster_center(face_oval)
    left_upper_lid = _mediapipe_points(dense, MEDIAPIPE_LEFT_UPPER_EYELID_INDICES)
    right_upper_lid = _mediapipe_points(dense, MEDIAPIPE_RIGHT_UPPER_EYELID_INDICES)
    left_lower_lid = _mediapipe_points(dense, MEDIAPIPE_LEFT_LOWER_EYELID_INDICES)
    right_lower_lid = _mediapipe_points(dense, MEDIAPIPE_RIGHT_LOWER_EYELID_INDICES)
    return {
        "dense": dense[:, :2],
        "face_oval": face_oval,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_upper_lid": left_upper_lid,
        "right_upper_lid": right_upper_lid,
        "left_lower_lid": left_lower_lid,
        "right_lower_lid": right_lower_lid,
        "left_brow": left_brow,
        "right_brow": right_brow,
        "lips_outer": lips_outer,
        "lips_inner": lips_inner,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "left_jaw": left_jaw,
        "right_jaw": right_jaw,
        "chin": chin,
        "mouth_left": mouth_left,
        "mouth_right": mouth_right,
        "mouth_top": mouth_top,
        "mouth_bottom": mouth_bottom,
        "upper_lip_top": upper_lip_top,
        "lower_lip_bottom": lower_lip_bottom,
        "upper_lip_inner": upper_lip_inner,
        "lower_lip_inner": lower_lip_inner,
        "cupid_left": cupid_left,
        "cupid_right": cupid_right,
        "nose_tip": nose_tip,
        "nose_bridge_top": nose_bridge_top,
        "nose_left": nose_left,
        "nose_right": nose_right,
        "nose_base_left": nose_base_left,
        "nose_base_right": nose_base_right,
        "left_eye_outer": left_eye_outer,
        "left_eye_inner": left_eye_inner,
        "right_eye_inner": right_eye_inner,
        "right_eye_outer": right_eye_outer,
        "left_eye_top": left_eye_top,
        "left_eye_bottom": left_eye_bottom,
        "right_eye_top": right_eye_top,
        "right_eye_bottom": right_eye_bottom,
        "left_brow_center": left_brow_center,
        "right_brow_center": right_brow_center,
        "mouth_center": mouth_center,
        "lip_center": lip_center,
        "face_center": face_center,
    }


def _mediapipe_raw_measurements(
    mediapipe_landmarks: list[list[float]] | None,
) -> dict[str, float | None]:
    mediapipe_result = empty_feature_dict([column for column in RAW_MEASUREMENT_COLUMNS if column.startswith("mediapipe_")])
    mediapipe_context = _derive_mediapipe_context(mediapipe_landmarks)
    if not mediapipe_context:
        return mediapipe_result

    face_oval = mediapipe_context.get("face_oval")
    left_eye_mp = mediapipe_context.get("left_eye")
    right_eye_mp = mediapipe_context.get("right_eye")
    left_upper_lid = mediapipe_context.get("left_upper_lid")
    right_upper_lid = mediapipe_context.get("right_upper_lid")
    left_lower_lid = mediapipe_context.get("left_lower_lid")
    right_lower_lid = mediapipe_context.get("right_lower_lid")
    left_brow_mp = mediapipe_context.get("left_brow")
    right_brow_mp = mediapipe_context.get("right_brow")
    mouth_left_mp = mediapipe_context.get("mouth_left")
    mouth_right_mp = mediapipe_context.get("mouth_right")
    mouth_top_mp = mediapipe_context.get("mouth_top")
    mouth_bottom_mp = mediapipe_context.get("mouth_bottom")
    upper_lip_top_mp = mediapipe_context.get("upper_lip_top")
    lower_lip_bottom_mp = mediapipe_context.get("lower_lip_bottom")
    upper_lip_inner_mp = mediapipe_context.get("upper_lip_inner")
    lower_lip_inner_mp = mediapipe_context.get("lower_lip_inner")
    cupid_left_mp = mediapipe_context.get("cupid_left")
    cupid_right_mp = mediapipe_context.get("cupid_right")
    nose_tip_mp = mediapipe_context.get("nose_tip")
    nose_bridge_top_mp = mediapipe_context.get("nose_bridge_top")
    nose_left_mp = mediapipe_context.get("nose_left")
    nose_right_mp = mediapipe_context.get("nose_right")
    nose_base_left_mp = mediapipe_context.get("nose_base_left")
    nose_base_right_mp = mediapipe_context.get("nose_base_right")
    left_eye_outer_mp = mediapipe_context.get("left_eye_outer")
    left_eye_inner_mp = mediapipe_context.get("left_eye_inner")
    right_eye_inner_mp = mediapipe_context.get("right_eye_inner")
    right_eye_outer_mp = mediapipe_context.get("right_eye_outer")
    left_eye_top_mp = mediapipe_context.get("left_eye_top")
    left_eye_bottom_mp = mediapipe_context.get("left_eye_bottom")
    right_eye_top_mp = mediapipe_context.get("right_eye_top")
    right_eye_bottom_mp = mediapipe_context.get("right_eye_bottom")
    left_cheek_mp = mediapipe_context.get("left_cheek")
    right_cheek_mp = mediapipe_context.get("right_cheek")
    left_jaw_mp = mediapipe_context.get("left_jaw")
    right_jaw_mp = mediapipe_context.get("right_jaw")
    chin_mp = mediapipe_context.get("chin")
    left_brow_center_mp = mediapipe_context.get("left_brow_center")
    right_brow_center_mp = mediapipe_context.get("right_brow_center")
    lip_center_mp = mediapipe_context.get("lip_center")
    face_center_mp = mediapipe_context.get("face_center")

    face_oval_width = _span(face_oval[:, 0]) if isinstance(face_oval, np.ndarray) and len(face_oval) else None
    face_oval_height = _span(face_oval[:, 1]) if isinstance(face_oval, np.ndarray) and len(face_oval) else None
    left_eye_width_mp = _distance(left_eye_outer_mp, left_eye_inner_mp)
    right_eye_width_mp = _distance(right_eye_inner_mp, right_eye_outer_mp)
    left_eye_height_mp = _distance(left_eye_top_mp, left_eye_bottom_mp)
    right_eye_height_mp = _distance(right_eye_top_mp, right_eye_bottom_mp)
    left_eye_area_mp = left_eye_width_mp * left_eye_height_mp if left_eye_width_mp is not None and left_eye_height_mp is not None else None
    right_eye_area_mp = right_eye_width_mp * right_eye_height_mp if right_eye_width_mp is not None and right_eye_height_mp is not None else None
    left_brow_width_mp = _span(left_brow_mp[:, 0]) if isinstance(left_brow_mp, np.ndarray) and len(left_brow_mp) else None
    right_brow_width_mp = _span(right_brow_mp[:, 0]) if isinstance(right_brow_mp, np.ndarray) and len(right_brow_mp) else None
    nose_width_mp = _distance(nose_left_mp, nose_right_mp)
    nose_base_width_mp = _distance(nose_base_left_mp, nose_base_right_mp)
    nose_length_mp = _distance(nose_bridge_top_mp, nose_tip_mp)
    dense = np.asarray(mediapipe_landmarks, dtype=np.float64) if mediapipe_landmarks else None
    nose_bridge_width_mp = _distance(_mediapipe_point(dense, 193), _mediapipe_point(dense, 417)) if dense is not None else None
    mouth_width_mp = _distance(mouth_left_mp, mouth_right_mp)
    mouth_height_mp = _distance(mouth_top_mp, mouth_bottom_mp)
    upper_lip_height_mp = _distance(upper_lip_top_mp, upper_lip_inner_mp)
    lower_lip_height_mp = _distance(lower_lip_inner_mp, lower_lip_bottom_mp)
    total_lip_height_mp = (
        float(upper_lip_height_mp + lower_lip_height_mp)
        if upper_lip_height_mp is not None and lower_lip_height_mp is not None
        else _distance(upper_lip_top_mp, lower_lip_bottom_mp)
    )
    upper_lip_area_mp = None
    lower_lip_area_mp = None
    if isinstance(mediapipe_context.get("lips_outer"), np.ndarray) and len(mediapipe_context.get("lips_outer")):
        lips_outer = np.asarray(mediapipe_context.get("lips_outer"), dtype=np.float64)
        lip_center_y = float(np.mean(lips_outer[:, 1]))
        upper_points = lips_outer[lips_outer[:, 1] <= lip_center_y]
        lower_points = lips_outer[lips_outer[:, 1] >= lip_center_y]
        upper_lip_area_mp = _polygon_area(upper_points)
        lower_lip_area_mp = _polygon_area(lower_points)
    if upper_lip_area_mp is None and mouth_width_mp is not None and upper_lip_height_mp is not None:
        upper_lip_area_mp = float(mouth_width_mp * upper_lip_height_mp)
    if lower_lip_area_mp is None and mouth_width_mp is not None and lower_lip_height_mp is not None:
        lower_lip_area_mp = float(mouth_width_mp * lower_lip_height_mp)
    cheekbone_width_mp = _distance(left_cheek_mp, right_cheek_mp)
    midface_width_mp = cheekbone_width_mp
    midface_height_mp = _distance(_mean_point([left_brow_center_mp, right_brow_center_mp]), lip_center_mp)
    jaw_width_mp = _distance(left_jaw_mp, right_jaw_mp)
    chin_width_mp = None
    if isinstance(face_oval, np.ndarray) and len(face_oval):
        chin_band = face_oval[face_oval[:, 1] >= np.quantile(face_oval[:, 1], 0.8)]
        if len(chin_band):
            chin_width_mp = _span(chin_band[:, 0])
    lower_face_contour_length_mp = None
    left_face_contour_length_mp = None
    right_face_contour_length_mp = None
    if isinstance(face_oval, np.ndarray) and len(face_oval) >= 6:
        chin_index = int(np.argmax(face_oval[:, 1]))
        left_arc = face_oval[: chin_index + 1]
        right_arc = face_oval[chin_index:]
        lower_arc = face_oval[max(chin_index - 4, 0) : min(chin_index + 5, len(face_oval))]
        left_face_contour_length_mp = _polyline_length(left_arc)
        right_face_contour_length_mp = _polyline_length(right_arc)
        lower_face_contour_length_mp = _polyline_length(lower_arc)

    mediapipe_result.update(
        {
            "mediapipe_face_oval_width": face_oval_width,
            "mediapipe_face_oval_height": face_oval_height,
            "mediapipe_left_eye_width": left_eye_width_mp,
            "mediapipe_right_eye_width": right_eye_width_mp,
            "mediapipe_left_eye_height": left_eye_height_mp,
            "mediapipe_right_eye_height": right_eye_height_mp,
            "mediapipe_left_eye_area_proxy": left_eye_area_mp,
            "mediapipe_right_eye_area_proxy": right_eye_area_mp,
            "mediapipe_left_upper_eyelid_curve_proxy": _curve_height_proxy(left_upper_lid, left_eye_outer_mp, left_eye_inner_mp),
            "mediapipe_right_upper_eyelid_curve_proxy": _curve_height_proxy(right_upper_lid, right_eye_inner_mp, right_eye_outer_mp),
            "mediapipe_left_lower_eyelid_curve_proxy": _curve_height_proxy(left_lower_lid, left_eye_outer_mp, left_eye_inner_mp),
            "mediapipe_right_lower_eyelid_curve_proxy": _curve_height_proxy(right_lower_lid, right_eye_inner_mp, right_eye_outer_mp),
            "mediapipe_left_brow_width": left_brow_width_mp,
            "mediapipe_right_brow_width": right_brow_width_mp,
            "mediapipe_left_brow_arch_height_proxy": _curve_height_proxy(left_brow_mp, left_brow_mp[0] if isinstance(left_brow_mp, np.ndarray) and len(left_brow_mp) else None, left_brow_mp[-1] if isinstance(left_brow_mp, np.ndarray) and len(left_brow_mp) else None),
            "mediapipe_right_brow_arch_height_proxy": _curve_height_proxy(right_brow_mp, right_brow_mp[0] if isinstance(right_brow_mp, np.ndarray) and len(right_brow_mp) else None, right_brow_mp[-1] if isinstance(right_brow_mp, np.ndarray) and len(right_brow_mp) else None),
            "mediapipe_nose_width": nose_width_mp,
            "mediapipe_nose_base_width": nose_base_width_mp,
            "mediapipe_nose_length": nose_length_mp,
            "mediapipe_nose_bridge_width_proxy": nose_bridge_width_mp,
            "mediapipe_nose_midline_offset": abs(float(nose_tip_mp[0] - face_center_mp[0])) if nose_tip_mp is not None and face_center_mp is not None else None,
            "mediapipe_mouth_width": mouth_width_mp,
            "mediapipe_mouth_height": mouth_height_mp,
            "mediapipe_upper_lip_height": upper_lip_height_mp,
            "mediapipe_lower_lip_height": lower_lip_height_mp,
            "mediapipe_total_lip_height": total_lip_height_mp,
            "mediapipe_upper_lip_area_proxy": upper_lip_area_mp,
            "mediapipe_lower_lip_area_proxy": lower_lip_area_mp,
            "mediapipe_cupid_bow_width_proxy": _distance(cupid_left_mp, cupid_right_mp),
            "mediapipe_cupid_bow_depth_proxy": _distance(upper_lip_top_mp, lip_center_mp),
            "mediapipe_cheekbone_width_proxy": cheekbone_width_mp,
            "mediapipe_midface_width_proxy": midface_width_mp,
            "mediapipe_midface_height_proxy": midface_height_mp,
            "mediapipe_jaw_width_proxy": jaw_width_mp,
            "mediapipe_chin_width_proxy": chin_width_mp,
            "mediapipe_chin_height_proxy": _distance(lip_center_mp, chin_mp),
            "mediapipe_lower_face_contour_length_proxy": lower_face_contour_length_mp,
            "mediapipe_left_face_contour_length_proxy": left_face_contour_length_mp,
            "mediapipe_right_face_contour_length_proxy": right_face_contour_length_mp,
            "mediapipe_chin_midline_offset": abs(float(chin_mp[0] - face_center_mp[0])) if chin_mp is not None and face_center_mp is not None else None,
        }
    )
    return mediapipe_result


def _feature_component(feature_name: str, family: str) -> str:
    normalized_name = feature_name.removeprefix("mediapipe_")
    if feature_name.startswith("sym_") or "symmetry" in feature_name or "midline" in feature_name:
        return "symmetry"
    if feature_name.startswith("angle_") or "pose_" in feature_name or "tilt" in feature_name or "axis" in feature_name:
        return "angles_pose"
    if "eye" in normalized_name or "canthal" in normalized_name:
        return "eyes"
    if "brow" in normalized_name:
        return "brows"
    if "nose" in normalized_name or "philtrum" in normalized_name:
        return "nose" if "mouth" not in normalized_name and "lip" not in normalized_name else "mouth_lips"
    if "mouth" in normalized_name or "lip" in normalized_name or "cupid" in normalized_name:
        return "mouth_lips"
    if "jaw" in normalized_name or "chin" in normalized_name or "lower_face" in normalized_name:
        return "jaw_chin"
    if "cheekbone" in normalized_name or "midface" in normalized_name:
        return "cheekbones"
    if "third" in normalized_name or "forehead" in normalized_name or "face_aspect" in normalized_name or "face_oval" in normalized_name:
        return "vertical_proportions" if "third" in normalized_name or "forehead" in normalized_name else "face_shape"
    if family == "normalized_ratios":
        return "horizontal_proportions"
    return "face_shape"


def _feature_units(feature_name: str, family: str) -> str:
    if family == "raw_measurements":
        return "px"
    if family == "angle_metrics":
        return "degrees"
    if family == "symmetry_metrics":
        return "index"
    return "ratio"


def _feature_source(feature_name: str, family: str) -> str:
    if feature_name.startswith("mediapipe_"):
        return "mediapipe_face_mesh"
    if family == "raw_measurements" and "proxy" not in feature_name:
        return "insightface_106"
    if family == "raw_measurements":
        return "fused"
    return "fused"


def _feature_reliability(feature_name: str, family: str, override: dict[str, Any]) -> str:
    explicit = override.get("reliability")
    if isinstance(explicit, str):
        return explicit
    if override.get("status") == "unstable":
        return "unstable"
    if feature_name.startswith("mediapipe_"):
        if family == "raw_measurements" and (
            "cheekbone" in feature_name
            or "jaw" in feature_name
            or "chin" in feature_name
            or "curve_proxy" in feature_name
            or "arch_height" in feature_name
            or "cupid" in feature_name
        ):
            return "proxy"
        if (
            "curve_proxy" in feature_name
            or "arch_height" in feature_name
            or "cupid" in feature_name
            or "prominence_proxy" in feature_name
            or "squareness_proxy" in feature_name
        ):
            return "proxy"
        return "reliable"
    if "proxy" in feature_name:
        return "proxy"
    if family in {"normalized_ratios", "symmetry_metrics", "angle_metrics"}:
        return "reliable"
    return "reliable"


def _feature_normalized_by(feature_name: str, family: str) -> str | None:
    if family != "normalized_ratios":
        return None
    suffix_map = {
        "_to_face_width": "raw_face_width_px",
        "_to_face_height": "raw_face_height_px",
        "_to_eye_width": "mean_eye_width",
        "_to_mouth_width": "raw_mouth_width_px",
        "_to_jaw_width": "jaw_width_proxy",
        "_to_lower_face_height": "lower_third_proxy",
    }
    for suffix, normalizer in suffix_map.items():
        if feature_name.endswith(suffix):
            return normalizer
    return "derived_ratio"


def _recommended_for_scorer(feature_name: str, family: str, reliability: str, override: dict[str, Any], status: str) -> tuple[bool, str | None]:
    if family == "raw_measurements":
        return False, "raw_pixel_feature"
    if reliability in {"unstable", "unavailable_without_extra_model"}:
        return False, reliability
    if reliability == "proxy":
        if feature_name.startswith("mediapipe_") and family in {"normalized_ratios", "symmetry_metrics", "angle_metrics"}:
            return True, None
        return False, "proxy"
    if status in {"redundant", "deprecated_but_kept_for_compatibility"}:
        return False, "compatibility_alias"
    if override.get("reference_inclusion") == "excluded":
        return False, "excluded_from_reference"
    return True, None


def parse_json_field(value: Any) -> Any:
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


def _distance(point_a: np.ndarray | None, point_b: np.ndarray | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(point_a - point_b))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _vector_angle_degrees(point_a: np.ndarray | None, point_b: np.ndarray | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    vector = point_b - point_a
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def _normalize_axis_angle_deg(angle: float | None) -> float | None:
    if angle is None:
        return None
    wrapped = (float(angle) + 90.0) % 180.0 - 90.0
    if wrapped == -90.0:
        return 90.0
    return float(wrapped)


def _minimal_axis_angle_difference_deg(angle_a: float | None, angle_b: float | None) -> float | None:
    if angle_a is None or angle_b is None:
        return None
    diff = abs(float(angle_a) - float(angle_b)) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return float(diff)


def _mirror_angle_asymmetry_deg(angle_left: float | None, angle_right: float | None) -> float | None:
    if angle_left is None or angle_right is None:
        return None
    return float(abs(abs(float(angle_left)) - abs(float(angle_right))))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not present:
        return None
    return float(sum(present) / len(present))


def _sum(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not present:
        return None
    return float(sum(present))


def _abs_diff(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(abs(value_a - value_b))


def _point_midpoint(point_a: np.ndarray | None, point_b: np.ndarray | None) -> np.ndarray | None:
    if point_a is None or point_b is None:
        return None
    return (point_a + point_b) / 2.0


def _mean_point(points: list[np.ndarray | None]) -> np.ndarray | None:
    present = [np.asarray(point, dtype=np.float64) for point in points if point is not None]
    if not present:
        return None
    return np.mean(np.vstack(present), axis=0)


def _point_from_xy(x: float | None, y: float | None) -> np.ndarray | None:
    if x is None or y is None:
        return None
    return np.asarray([float(x), float(y)], dtype=np.float64)


def _span(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values) - np.min(values))


def _clip_box_to_image(box: list[float], image_width: float | None, image_height: float | None) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    if image_width is not None:
        x1 = max(0.0, min(x1, float(image_width)))
        x2 = max(0.0, min(x2, float(image_width)))
    if image_height is not None:
        y1 = max(0.0, min(y1, float(image_height)))
        y2 = max(0.0, min(y2, float(image_height)))
    return [float(x1), float(y1), float(max(x1, x2)), float(max(y1, y2))]


def empty_feature_dict(columns: list[str]) -> dict[str, float | None]:
    return {column: None for column in columns}


def _region_points(
    points: np.ndarray | None,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = np.ones((points.shape[0],), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    return points[mask]


def _indexed_region_points(
    points: np.ndarray | None,
    *,
    indices: np.ndarray | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float64)
    point_indices = np.arange(points.shape[0], dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32)
    mask = np.ones((points.shape[0],), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    return point_indices[mask], points[mask]


def _extreme_point(points: np.ndarray, *, axis: int, mode: str) -> np.ndarray | None:
    if len(points) == 0:
        return None
    index = int(np.argmin(points[:, axis])) if mode == "min" else int(np.argmax(points[:, axis]))
    return points[index]


def _cluster_center(points: np.ndarray) -> np.ndarray | None:
    if len(points) == 0:
        return None
    return points.mean(axis=0)


def _closest_point_to_x(points: np.ndarray, target_x: float, *, mode_y: str | None = None) -> np.ndarray | None:
    if len(points) == 0:
        return None
    x_scores = np.abs(points[:, 0] - target_x)
    if mode_y == "min":
        y_scores = points[:, 1] - np.min(points[:, 1])
        scores = x_scores + 0.1 * y_scores
    elif mode_y == "max":
        y_scores = np.max(points[:, 1]) - points[:, 1]
        scores = x_scores + 0.1 * y_scores
    else:
        scores = x_scores
    return points[int(np.argmin(scores))]


def _indexed_closest_point_to_x(
    points: np.ndarray,
    indices: np.ndarray,
    target_x: float,
    *,
    mode_y: str | None = None,
) -> tuple[np.ndarray | None, list[int]]:
    if len(points) == 0:
        return None, []
    x_scores = np.abs(points[:, 0] - target_x)
    if mode_y == "min":
        y_scores = points[:, 1] - np.min(points[:, 1])
        scores = x_scores + 0.1 * y_scores
    elif mode_y == "max":
        y_scores = np.max(points[:, 1]) - points[:, 1]
        scores = x_scores + 0.1 * y_scores
    else:
        scores = x_scores
    winner = int(np.argmin(scores))
    return points[winner], [int(indices[winner])]


def _semantic_point_debug_entry(
    *,
    point_name: str,
    point: np.ndarray | None,
    group: str,
    source_type: str,
    source_indices: list[int] | None,
    description: str,
    source_region: str | None = None,
    validation_status: str | None = None,
) -> dict[str, Any] | None:
    if point is None:
        return None
    normalized_indices = [int(value) for value in (source_indices or [])]
    source_index_106 = normalized_indices[0] if len(normalized_indices) == 1 else None
    return {
        "point_name": point_name,
        "group": group,
        "x": float(point[0]),
        "y": float(point[1]),
        "source_type": source_type,
        "source_indices": normalized_indices,
        "source_index_106": source_index_106,
        "source_region": source_region or (
            region_name_for_index(source_index_106) if source_index_106 is not None else None
        ),
        "validation_status": validation_status or ("fallback" if "fallback" in source_type else "validated"),
        "description": description,
    }


def _safe_index_point(points: np.ndarray | None, index: int) -> np.ndarray | None:
    if points is None:
        return None
    if index < 0 or index >= len(points):
        return None
    point = np.asarray(points[index], dtype=np.float64)
    if point.shape != (2,):
        return None
    return point


def _semantic_point_payload(
    *,
    point_name: str,
    point: np.ndarray | None,
    source_type: str,
    source_indices: list[int] | None,
    source_region: str | None,
    validation_status: str,
    source_rule: str | None = None,
) -> dict[str, Any] | None:
    if point is None:
        return None
    normalized_indices = [int(value) for value in (source_indices or [])]
    source_index_106 = normalized_indices[0] if len(normalized_indices) == 1 else None
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "source_index_106": source_index_106,
        "source_indices": normalized_indices,
        "source_region": source_region,
        "source_type": source_type,
        "validation_status": validation_status,
        "point_name": point_name,
        "source_rule": source_rule,
    }


def _region_for_source_indices(source_indices: list[int] | None) -> str | None:
    normalized_indices = [int(value) for value in (source_indices or [])]
    if not normalized_indices:
        return None
    regions = {region_name_for_index(index) for index in normalized_indices}
    if len(regions) == 1:
        return next(iter(regions))
    return "mixed"


def _indexed_extreme_point(
    points: np.ndarray,
    indices: np.ndarray,
    *,
    axis: int,
    mode: str,
) -> tuple[np.ndarray | None, list[int]]:
    if len(points) == 0 or len(indices) == 0:
        return None, []
    winner = int(np.argmin(points[:, axis])) if mode == "min" else int(np.argmax(points[:, axis]))
    return points[winner], [int(indices[winner])]


def _make_semantic_meta(
    *,
    point_name: str,
    point: np.ndarray | None,
    group: str,
    source_type: str,
    source_indices: list[int] | None,
    description: str,
    source_region: str | None = None,
    validation_status: str | None = None,
    source_rule: str | None = None,
) -> dict[str, Any] | None:
    entry = _semantic_point_debug_entry(
        point_name=point_name,
        point=point,
        group=group,
        source_type=source_type,
        source_indices=source_indices,
        description=description,
        source_region=source_region or _region_for_source_indices(source_indices),
        validation_status=validation_status,
    )
    if entry is not None:
        entry["source_rule"] = source_rule
    return entry


def _is_valid_mouth_pair(
    *,
    left_mouth: np.ndarray | None,
    right_mouth: np.ndarray | None,
    nose_tip: np.ndarray,
    bbox: list[float],
) -> bool:
    if left_mouth is None or right_mouth is None:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    mouth_width = float(right_mouth[0] - left_mouth[0])
    if mouth_width <= 0.0:
        return False
    if mouth_width < 0.08 * face_width or mouth_width > 0.70 * face_width:
        return False
    if float(left_mouth[1]) <= float(nose_tip[1]) + 0.01 * face_height:
        return False
    if float(right_mouth[1]) <= float(nose_tip[1]) + 0.01 * face_height:
        return False
    lower_half_top = y1 + 0.52 * face_height
    if float(left_mouth[1]) < lower_half_top or float(right_mouth[1]) < lower_half_top:
        return False
    if (
        abs(float(left_mouth[0] - face_center_x)) > 0.40 * face_width
        or abs(float(right_mouth[0] - face_center_x)) > 0.40 * face_width
    ):
        return False
    return True


def _derive_mouth_geometry_from_106_indices(
    *,
    dense: np.ndarray | None,
    bbox: list[float],
    nose_tip: np.ndarray,
) -> dict[str, Any] | None:
    if dense is None or not has_106_landmarks(dense):
        return None
    left_idx = int(MOUTH_LEFT_CORNER_INDEX)
    right_idx = int(MOUTH_RIGHT_CORNER_INDEX)
    upper_idx = int(UPPER_LIP_CENTER_TOP_INDEX)
    lower_idx = int(LOWER_LIP_CENTER_BOTTOM_INDEX)
    left_mouth = _safe_index_point(dense, left_idx)
    right_mouth = _safe_index_point(dense, right_idx)
    if left_mouth is None or right_mouth is None:
        return None
    mouth_center = (left_mouth + right_mouth) / 2.0
    upper_lip_top = _safe_index_point(dense, upper_idx)
    lower_lip_bottom = _safe_index_point(dense, lower_idx)
    return {
        "left_mouth": left_mouth,
        "right_mouth": right_mouth,
        "mouth_center": mouth_center,
        "upper_lip_top": upper_lip_top,
        "lower_lip_bottom": lower_lip_bottom,
        "source_type": "insightface_106_indices",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=left_mouth,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[left_idx],
                source_region="mouth",
                description="Left lip corner from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=right_mouth,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[right_idx],
                source_region="mouth",
                description="Right lip corner from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=mouth_center,
                group="mouth",
                source_type="derived",
                source_indices=[left_idx, right_idx],
                source_region="mouth",
                description="Midpoint between indexed mouth corners.",
            ),
            "upper_lip_top": _semantic_point_debug_entry(
                point_name="upper_lip_top",
                point=upper_lip_top,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[upper_idx],
                source_region="mouth",
                description="Upper-lip apex from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "lower_lip_bottom": _semantic_point_debug_entry(
                point_name="lower_lip_bottom",
                point=lower_lip_bottom,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[lower_idx],
                source_region="mouth",
                description="Lower-lip nadir from the canonical InsightFace landmark_2d_106 index map.",
            ),
        },
    }


def _derive_mouth_geometry_from_dense(
    *,
    dense: np.ndarray | None,
    bbox: list[float] | None,
    nose_tip: np.ndarray,
    fallback_left: np.ndarray,
    fallback_right: np.ndarray,
) -> dict[str, Any]:
    fallback_center = (fallback_left + fallback_right) / 2.0
    fallback = {
        "left_mouth": fallback_left,
        "right_mouth": fallback_right,
        "mouth_center": fallback_center,
        "upper_lip_top": None,
        "lower_lip_bottom": None,
        "source_type": "anchor_fallback",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=fallback_left,
                group="mouth",
                source_type="anchor_fallback",
                source_indices=[3],
                description="Fallback mouth corner from five-point anchor landmarks.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=fallback_right,
                group="mouth",
                source_type="anchor_fallback",
                source_indices=[4],
                description="Fallback mouth corner from five-point anchor landmarks.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=fallback_center,
                group="mouth",
                source_type="derived",
                source_indices=[3, 4],
                description="Midpoint between fallback mouth corner anchors.",
            ),
        },
    }
    if dense is None or bbox is None or len(dense) == 0 or len(bbox) != 4:
        return fallback

    if has_106_landmarks(dense):
        indexed_geometry = _derive_mouth_geometry_from_106_indices(
            dense=dense,
            bbox=bbox,
            nose_tip=nose_tip,
        )
        if indexed_geometry is not None:
            return indexed_geometry

    x1, y1, x2, _y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(float(bbox[3]) - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    candidate_dense = dense
    candidate_indices = np.arange(dense.shape[0], dtype=np.int32)
    if has_106_landmarks(dense):
        candidate_indices, candidate_dense = points_for_region(dense, "mouth")

    region_indices, mouth_region = _indexed_region_points(
        candidate_dense,
        indices=candidate_indices,
        x_min=float(face_center_x - 0.32 * face_width),
        x_max=float(face_center_x + 0.32 * face_width),
        y_min=float(max(nose_tip[1] + 0.06 * face_height, y1 + 0.58 * face_height)),
        y_max=float(y1 + 0.84 * face_height),
    )
    if len(mouth_region) < 6:
        region_indices, mouth_region = _indexed_region_points(
            candidate_dense,
            indices=candidate_indices,
            x_min=float(face_center_x - 0.40 * face_width),
            x_max=float(face_center_x + 0.40 * face_width),
            y_min=float(max(nose_tip[1] + 0.04 * face_height, y1 + 0.54 * face_height)),
            y_max=float(y1 + 0.88 * face_height),
        )
    if len(mouth_region) < 4:
        return fallback

    y_low = float(np.quantile(mouth_region[:, 1], 0.20))
    y_high = float(np.quantile(mouth_region[:, 1], 0.80))
    band_indices, lip_band = _indexed_region_points(
        mouth_region,
        indices=region_indices,
        y_min=y_low,
        y_max=y_high,
    )
    if len(lip_band) < 4:
        band_indices, lip_band = region_indices, mouth_region

    band_center_y = float(np.median(lip_band[:, 1]))
    left_indices, left_candidates = _indexed_region_points(
        lip_band,
        indices=band_indices,
        x_max=face_center_x,
    )
    right_indices, right_candidates = _indexed_region_points(
        lip_band,
        indices=band_indices,
        x_min=face_center_x,
    )
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return fallback

    target_left_x = float(np.quantile(lip_band[:, 0], 0.15))
    target_right_x = float(np.quantile(lip_band[:, 0], 0.85))
    x_scale = max(1.0, 0.25 * face_width)
    y_scale = max(1.0, 0.25 * face_height)
    left_scores = (
        np.abs(left_candidates[:, 0] - target_left_x) / x_scale
        + np.abs(left_candidates[:, 1] - band_center_y) / y_scale
    )
    right_scores = (
        np.abs(right_candidates[:, 0] - target_right_x) / x_scale
        + np.abs(right_candidates[:, 1] - band_center_y) / y_scale
    )
    left_pick = int(np.argmin(left_scores))
    right_pick = int(np.argmin(right_scores))
    left_mouth = left_candidates[left_pick]
    right_mouth = right_candidates[right_pick]
    left_source = [int(left_indices[left_pick])]
    right_source = [int(right_indices[right_pick])]

    if not _is_valid_mouth_pair(
        left_mouth=left_mouth,
        right_mouth=right_mouth,
        nose_tip=nose_tip,
        bbox=bbox,
    ):
        return fallback

    mouth_center = (left_mouth + right_mouth) / 2.0
    central_indices, central_lip_region = _indexed_region_points(
        mouth_region,
        indices=region_indices,
        x_min=float(mouth_center[0] - 0.10 * face_width),
        x_max=float(mouth_center[0] + 0.10 * face_width),
    )
    if len(central_lip_region) < 2:
        central_indices, central_lip_region = _indexed_region_points(
            mouth_region,
            indices=region_indices,
            x_min=float(face_center_x - 0.12 * face_width),
            x_max=float(face_center_x + 0.12 * face_width),
        )

    upper_lip_top, upper_source = _indexed_closest_point_to_x(
        central_lip_region,
        central_indices,
        float(mouth_center[0]),
        mode_y="min",
    )
    lower_lip_bottom, lower_source = _indexed_closest_point_to_x(
        central_lip_region,
        central_indices,
        float(mouth_center[0]),
        mode_y="max",
    )

    return {
        "left_mouth": left_mouth,
        "right_mouth": right_mouth,
        "mouth_center": mouth_center,
        "upper_lip_top": upper_lip_top,
        "lower_lip_bottom": lower_lip_bottom,
        "source_type": "heuristic_dense_lips",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=left_mouth,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=left_source,
                description="Left lip corner selected from the central lip band of the dense landmark set.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=right_mouth,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=right_source,
                description="Right lip corner selected from the central lip band of the dense landmark set.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=mouth_center,
                group="mouth",
                source_type="derived",
                source_indices=left_source + right_source,
                description="Midpoint between the selected lip corners.",
            ),
            "upper_lip_top": _semantic_point_debug_entry(
                point_name="upper_lip_top",
                point=upper_lip_top,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=upper_source,
                description="Highest central lip point near the mouth midline.",
            ),
            "lower_lip_bottom": _semantic_point_debug_entry(
                point_name="lower_lip_bottom",
                point=lower_lip_bottom,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=lower_source,
                description="Lowest central lip point near the mouth midline.",
            ),
        },
    }


def _validate_semantic_geometry(
    *,
    bbox: list[float] | None,
    left_eye: np.ndarray | None,
    right_eye: np.ndarray | None,
    nose_tip: np.ndarray | None,
    nose_base_left: np.ndarray | None,
    nose_base_right: np.ndarray | None,
    mouth_left: np.ndarray | None,
    mouth_right: np.ndarray | None,
    mouth_center: np.ndarray | None,
    chin_center: np.ndarray | None,
    jaw_left: np.ndarray | None,
    jaw_right: np.ndarray | None,
    cheekbone_left: np.ndarray | None,
    cheekbone_right: np.ndarray | None,
    semantic_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    failed_checks: list[str] = []
    if bbox is None or len(bbox) != 4:
        warnings.append("semantic_validation_skipped_missing_bbox")
        return {
            "passed": False,
            "warnings": warnings,
            "failed_checks": ["missing_bbox"],
            "unreliable_features": sorted(SEMANTIC_UNRELIABLE_RAW_FEATURES),
        }

    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    lower_half_top = y1 + 0.50 * face_height

    def _meta(name: str) -> dict[str, Any]:
        if semantic_metadata is None:
            return {}
        value = semantic_metadata.get(name)
        return value if isinstance(value, dict) else {}

    mouth_left_meta = _meta("left_mouth")
    mouth_right_meta = _meta("right_mouth")
    nose_tip_meta = _meta("nose_tip")
    nose_base_left_meta = _meta("nose_base_left")
    nose_base_right_meta = _meta("nose_base_right")
    left_eye_meta = _meta("left_eye_center")
    right_eye_meta = _meta("right_eye_center")

    if mouth_left_meta.get("source_indices") and mouth_left_meta.get("source_indices") != [MOUTH_LEFT_CORNER_INDEX]:
        failed_checks.append("mouth_left_wrong_source_index")
    if mouth_right_meta.get("source_indices") and mouth_right_meta.get("source_indices") != [MOUTH_RIGHT_CORNER_INDEX]:
        failed_checks.append("mouth_right_wrong_source_index")
    if mouth_left_meta.get("source_region") not in {None, "mouth"}:
        failed_checks.append("mouth_left_wrong_source_region")
    if mouth_right_meta.get("source_region") not in {None, "mouth"}:
        failed_checks.append("mouth_right_wrong_source_region")

    if nose_tip_meta.get("source_indices") and nose_tip_meta.get("source_indices") != [NOSE_TIP_INDEX]:
        failed_checks.append("nose_tip_wrong_source_index")
    if nose_tip_meta.get("source_region") not in {None, "nose"}:
        failed_checks.append("nose_tip_wrong_source_region")

    for name, meta, expected in (
        ("nose_base_left", nose_base_left_meta, [NOSE_BASE_LEFT_INDEX]),
        ("nose_base_right", nose_base_right_meta, [NOSE_BASE_RIGHT_INDEX]),
    ):
        source_indices = [int(value) for value in meta.get("source_indices", [])]
        if source_indices and any(index in {MOUTH_LEFT_CORNER_INDEX, MOUTH_RIGHT_CORNER_INDEX} for index in source_indices):
            failed_checks.append(f"{name}_uses_mouth_indices")
        if source_indices and any(index < 77 or index > 85 for index in source_indices):
            failed_checks.append(f"{name}_outside_nose_base_region")
        if source_indices and source_indices != expected:
            failed_checks.append(f"{name}_wrong_source_index")
        if meta.get("source_region") not in {None, "nose"}:
            failed_checks.append(f"{name}_wrong_source_region")

    for name, meta, allowed_indices in (
        ("left_eye_center", left_eye_meta, set(LEFT_EYE_INDICES)),
        ("right_eye_center", right_eye_meta, set(RIGHT_EYE_INDICES)),
    ):
        source_indices = [int(value) for value in meta.get("source_indices", [])]
        if source_indices and any(index not in allowed_indices for index in source_indices):
            failed_checks.append(f"{name}_outside_eye_cluster")

    if mouth_left is None or mouth_right is None or mouth_center is None:
        failed_checks.append("missing_mouth_points")
    if nose_tip is None:
        failed_checks.append("missing_nose_tip")

    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    if mouth_left is not None and mouth_right is not None:
        if not (float(mouth_left[0]) < float(mouth_right[0])):
            failed_checks.append("mouth_left_not_left_of_right")
        mouth_width = float(abs(mouth_right[0] - mouth_left[0]))
        if mouth_width < 0.08 * face_width or mouth_width > 0.70 * face_width:
            failed_checks.append("mouth_width_out_of_range")
        if float(mouth_left[1]) < lower_half_top or float(mouth_right[1]) < lower_half_top:
            failed_checks.append("mouth_corners_above_lower_half")
        if (
            abs(float(mouth_left[0] - face_center_x)) > 0.46 * face_width
            or abs(float(mouth_right[0] - face_center_x)) > 0.46 * face_width
        ):
            failed_checks.append("mouth_corners_far_from_center")

    if mouth_left is not None and mouth_right is not None and mouth_center is not None:
        if not (float(mouth_left[0]) <= float(mouth_center[0]) <= float(mouth_right[0])):
            failed_checks.append("mouth_center_not_between_corners")

    if nose_tip is not None and mouth_left is not None and mouth_right is not None:
        if float(mouth_left[1]) <= float(nose_tip[1]) or float(mouth_right[1]) <= float(nose_tip[1]):
            failed_checks.append("mouth_corners_above_nose_tip")
    if nose_base_center is not None and mouth_left is not None and mouth_right is not None:
        if float(mouth_left[1]) <= float(nose_base_center[1]) or float(mouth_right[1]) <= float(nose_base_center[1]):
            failed_checks.append("mouth_corners_above_nose_base")

    if left_eye is not None and right_eye is not None and nose_tip is not None and mouth_center is not None:
        eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
        if not (eye_center_y < float(nose_tip[1]) < float(mouth_center[1])):
            failed_checks.append("nose_tip_not_between_eyes_and_mouth")

    if nose_base_center is not None and left_eye is not None and right_eye is not None:
        eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
        if float(nose_base_center[1]) <= eye_center_y:
            failed_checks.append("nose_base_above_eyes")
        if mouth_center is not None and float(nose_base_center[1]) >= float(mouth_center[1]):
            failed_checks.append("nose_base_below_mouth")

    if chin_center is not None and mouth_center is not None and float(chin_center[1]) <= float(mouth_center[1]):
        failed_checks.append("chin_not_below_mouth")
    if jaw_left is not None and mouth_center is not None and float(jaw_left[1]) <= float(mouth_center[1]):
        failed_checks.append("jaw_left_not_below_mouth")
    if jaw_right is not None and mouth_center is not None and float(jaw_right[1]) <= float(mouth_center[1]):
        failed_checks.append("jaw_right_not_below_mouth")

    if mouth_left is not None and jaw_left is not None:
        if float(np.linalg.norm(mouth_left - jaw_left)) < 0.08 * face_width:
            failed_checks.append("mouth_left_too_close_to_jaw_left")
    if mouth_right is not None and jaw_right is not None:
        if float(np.linalg.norm(mouth_right - jaw_right)) < 0.08 * face_width:
            failed_checks.append("mouth_right_too_close_to_jaw_right")
    if mouth_left is not None and cheekbone_left is not None:
        if float(np.linalg.norm(mouth_left - cheekbone_left)) < 0.07 * face_width:
            failed_checks.append("mouth_left_too_close_to_cheekbone_left")
    if mouth_right is not None and cheekbone_right is not None:
        if float(np.linalg.norm(mouth_right - cheekbone_right)) < 0.07 * face_width:
            failed_checks.append("mouth_right_too_close_to_cheekbone_right")

    warning_checks = sorted({check for check in failed_checks if check in SEMANTIC_WARNING_CHECKS})
    critical_failed_checks = sorted({check for check in failed_checks if check not in SEMANTIC_WARNING_CHECKS})
    passed = len(critical_failed_checks) == 0
    if warning_checks:
        warnings.extend(f"semantic_warning:{check}" for check in warning_checks)
    if critical_failed_checks:
        warnings.append("semantic_validation_failed")
    return {
        "passed": passed,
        "warnings": warnings,
        "failed_checks": critical_failed_checks,
        "warning_checks": warning_checks,
        "unreliable_features": (
            sorted(SEMANTIC_WARNING_UNRELIABLE_FEATURES)
            if passed and warning_checks
            else (
                []
                if passed
                else sorted(SEMANTIC_UNRELIABLE_RAW_FEATURES.union(SEMANTIC_UNRELIABLE_ANGLE_FEATURES))
            )
        ),
    }


def _build_semantic_face_extents(
    *,
    contour_points: np.ndarray,
    left_contour_points: np.ndarray,
    right_contour_points: np.ndarray,
    left_eye_center: np.ndarray | None,
    right_eye_center: np.ndarray | None,
    brow_center: np.ndarray | None,
    nose_base_center: np.ndarray | None,
    chin_center: np.ndarray | None,
    jaw_left: np.ndarray | None,
    jaw_right: np.ndarray | None,
    cheekbone_left: np.ndarray | None,
    cheekbone_right: np.ndarray | None,
    detector_bbox: list[float] | None,
) -> dict[str, Any] | None:
    candidate_x_left: list[float] = []
    candidate_x_right: list[float] = []
    candidate_y_top: list[float] = []
    candidate_y_bottom: list[float] = []

    if len(left_contour_points) > 0:
        candidate_x_left.append(float(np.min(left_contour_points[:, 0])))
        candidate_y_top.append(float(np.min(left_contour_points[:, 1])))
    if len(right_contour_points) > 0:
        candidate_x_right.append(float(np.max(right_contour_points[:, 0])))
        candidate_y_top.append(float(np.min(right_contour_points[:, 1])))
    if len(contour_points) > 0:
        candidate_y_bottom.append(float(np.max(contour_points[:, 1])))

    for point in (jaw_left, cheekbone_left):
        if point is not None:
            candidate_x_left.append(float(point[0]))
    for point in (jaw_right, cheekbone_right):
        if point is not None:
            candidate_x_right.append(float(point[0]))
    for point in (chin_center, jaw_left, jaw_right):
        if point is not None:
            candidate_y_bottom.append(float(point[1]))

    brow_to_eye = None
    if brow_center is not None and left_eye_center is not None and right_eye_center is not None:
        eye_center_y = float((left_eye_center[1] + right_eye_center[1]) / 2.0)
        brow_to_eye = abs(eye_center_y - float(brow_center[1]))
        forehead_proxy_top = float(brow_center[1]) - max(brow_to_eye * 1.15, 1.0)
        candidate_y_top.append(forehead_proxy_top)
    elif brow_center is not None and nose_base_center is not None:
        forehead_proxy_top = float(brow_center[1]) - max(abs(float(nose_base_center[1] - brow_center[1])) * 0.55, 1.0)
        candidate_y_top.append(forehead_proxy_top)

    if not candidate_x_left or not candidate_x_right or not candidate_y_top or not candidate_y_bottom:
        return None

    left = min(candidate_x_left)
    right = max(candidate_x_right)
    top = min(candidate_y_top)
    bottom = max(candidate_y_bottom)
    if detector_bbox and len(detector_bbox) == 4:
        _, detector_top, _, _ = [float(value) for value in detector_bbox]
        top = min(top, detector_top)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    return {
        "left": float(left),
        "right": float(right),
        "top": float(top),
        "bottom": float(bottom),
        "width": float(width),
        "height": float(height),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "source_type": "semantic_face_extents",
        "source_rule": "contour_x_extrema + jaw/cheek support + forehead proxy from brow-eye geometry + chin/contour bottom",
        "source_indices": sorted(set(int(index) for index in CONTOUR_INDICES)),
    }


def _derive_dense_context_from_insightface_106(
    *,
    landmarks: list[list[float]],
    landmarks_dense: list[list[float]],
    bbox: list[float],
) -> dict[str, np.ndarray | float | None]:
    context: dict[str, np.ndarray | float | None] = {}
    points5 = np.asarray(landmarks[:5], dtype=np.float64)
    left_eye_5, right_eye_5, nose_tip_5, left_mouth_5, right_mouth_5 = points5
    dense = np.asarray(landmarks_dense, dtype=np.float64)
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    semantic_meta: dict[str, dict[str, Any]] = {}

    def register(
        name: str,
        point: np.ndarray | None,
        *,
        group: str,
        source_type: str,
        source_indices: list[int] | None,
        description: str,
        source_rule: str | None = None,
    ) -> None:
        if point is None:
            return
        context[name] = point
        normalized_indices = [int(value) for value in (source_indices or [])]
        context[f"{name}_source_indices"] = normalized_indices
        meta = _make_semantic_meta(
            point_name=name,
            point=point,
            group=group,
            source_type=source_type,
            source_indices=normalized_indices,
            source_region=_region_for_source_indices(normalized_indices) or group,
            description=description,
            validation_status="validated",
            source_rule=source_rule,
        )
        if meta is not None:
            semantic_meta[name] = meta

    left_eye_indices, left_eye_points = points_for_region(dense, "left_eye")
    right_eye_indices, right_eye_points = points_for_region(dense, "right_eye")
    left_brow_indices, left_brow_points = points_for_region(dense, "left_brow")
    right_brow_indices, right_brow_points = points_for_region(dense, "right_brow")
    left_contour_indices, left_contour_points = points_for_indices(dense, LEFT_CONTOUR_INDICES)
    right_contour_indices, right_contour_points = points_for_indices(dense, RIGHT_CONTOUR_INDICES)
    contour_indices, contour_points = points_for_indices(dense, CONTOUR_INDICES)
    upper_lip_arc_indices, upper_lip_arc_points = points_for_indices(dense, UPPER_LIP_ARC_INDICES)

    left_eye_outer, left_eye_outer_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=0, mode="min")
    left_eye_inner, left_eye_inner_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=0, mode="max")
    right_eye_inner, right_eye_inner_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=0, mode="min")
    right_eye_outer, right_eye_outer_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=0, mode="max")
    left_eye_top, left_eye_top_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=1, mode="min")
    left_eye_bottom, left_eye_bottom_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=1, mode="max")
    right_eye_top, right_eye_top_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=1, mode="min")
    right_eye_bottom, right_eye_bottom_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=1, mode="max")

    left_eye_center = left_eye_points.mean(axis=0) if len(left_eye_points) > 0 else left_eye_5
    right_eye_center = right_eye_points.mean(axis=0) if len(right_eye_points) > 0 else right_eye_5
    eye_center = _point_midpoint(left_eye_center, right_eye_center)
    left_brow = left_brow_points.mean(axis=0) if len(left_brow_points) > 0 else None
    right_brow = right_brow_points.mean(axis=0) if len(right_brow_points) > 0 else None
    brow_center = _point_midpoint(left_brow, right_brow)

    nose_tip = point_for_index(dense, NOSE_TIP_INDEX)
    if nose_tip is None:
        nose_tip = nose_tip_5
    nose_bridge_top = point_for_index(dense, int(NOSE_BRIDGE_INDICES[0]))
    alare_left = point_for_index(dense, ALARE_LEFT_INDEX)
    nose_base_left = point_for_index(dense, NOSE_BASE_LEFT_INDEX)
    subnasale_left = point_for_index(dense, SUBNASALE_LEFT_INDEX)
    subnasale_center = point_for_index(dense, SUBNASALE_CENTER_INDEX)
    alare_right = point_for_index(dense, ALARE_RIGHT_INDEX)
    nose_base_right = point_for_index(dense, NOSE_BASE_RIGHT_INDEX)
    subnasale_right = point_for_index(dense, SUBNASALE_RIGHT_INDEX)
    nose_mid_left = alare_left
    nose_mid_right = alare_right
    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    mouth_geometry = _derive_mouth_geometry_from_106_indices(
        dense=dense,
        bbox=bbox,
        nose_tip=nose_tip if nose_tip is not None else nose_tip_5,
    ) or {}
    left_mouth = mouth_geometry.get("left_mouth") if isinstance(mouth_geometry.get("left_mouth"), np.ndarray) else left_mouth_5
    right_mouth = (
        mouth_geometry.get("right_mouth") if isinstance(mouth_geometry.get("right_mouth"), np.ndarray) else right_mouth_5
    )
    mouth_center = (
        mouth_geometry.get("mouth_center") if isinstance(mouth_geometry.get("mouth_center"), np.ndarray) else _point_midpoint(left_mouth, right_mouth)
    )
    upper_lip_top = (
        mouth_geometry.get("upper_lip_top")
        if isinstance(mouth_geometry.get("upper_lip_top"), np.ndarray)
        else point_for_index(dense, UPPER_LIP_CENTER_TOP_INDEX)
    )
    lower_lip_bottom = (
        mouth_geometry.get("lower_lip_bottom")
        if isinstance(mouth_geometry.get("lower_lip_bottom"), np.ndarray)
        else point_for_index(dense, LOWER_LIP_CENTER_BOTTOM_INDEX)
    )
    philtrum_top = upper_lip_top
    upper_lip_left_shoulder = point_for_index(dense, 63)
    upper_lip_right_shoulder = point_for_index(dense, 67)
    lip_midline_center = _point_midpoint(upper_lip_top, lower_lip_bottom)

    contour_center_x = face_center_x if mouth_center is None else float(mouth_center[0])
    lower_left_band = left_contour_points[left_contour_points[:, 1] >= np.quantile(left_contour_points[:, 1], 0.55)] if len(left_contour_points) > 0 else np.zeros((0, 2), dtype=np.float64)
    lower_left_indices = left_contour_indices[left_contour_points[:, 1] >= np.quantile(left_contour_points[:, 1], 0.55)] if len(left_contour_points) > 0 else np.zeros((0,), dtype=np.int32)
    lower_right_band = right_contour_points[right_contour_points[:, 1] >= np.quantile(right_contour_points[:, 1], 0.55)] if len(right_contour_points) > 0 else np.zeros((0, 2), dtype=np.float64)
    lower_right_indices = right_contour_indices[right_contour_points[:, 1] >= np.quantile(right_contour_points[:, 1], 0.55)] if len(right_contour_points) > 0 else np.zeros((0,), dtype=np.int32)
    jaw_left, jaw_left_indices = _indexed_extreme_point(lower_left_band, lower_left_indices, axis=0, mode="min")
    jaw_right, jaw_right_indices = _indexed_extreme_point(lower_right_band, lower_right_indices, axis=0, mode="max")

    cheek_low = y1 + 0.32 * face_height
    cheek_high = y1 + 0.62 * face_height
    cheek_left_mask = (left_contour_points[:, 1] >= cheek_low) & (left_contour_points[:, 1] <= cheek_high) if len(left_contour_points) > 0 else np.zeros((0,), dtype=bool)
    cheek_right_mask = (right_contour_points[:, 1] >= cheek_low) & (right_contour_points[:, 1] <= cheek_high) if len(right_contour_points) > 0 else np.zeros((0,), dtype=bool)
    cheekbone_left, cheekbone_left_indices = _indexed_extreme_point(left_contour_points[cheek_left_mask], left_contour_indices[cheek_left_mask], axis=0, mode="min")
    cheekbone_right, cheekbone_right_indices = _indexed_extreme_point(right_contour_points[cheek_right_mask], right_contour_indices[cheek_right_mask], axis=0, mode="max")
    chin_center = point_for_index(dense, 0)

    chin_band_mask = (
        (contour_points[:, 1] >= y1 + 0.78 * face_height)
        & (contour_points[:, 1] <= y2)
        & (np.abs(contour_points[:, 0] - contour_center_x) <= 0.22 * face_width)
    ) if len(contour_points) > 0 else np.zeros((0,), dtype=bool)
    chin_band_points = contour_points[chin_band_mask]
    philtrum_width_proxy = _distance(upper_lip_left_shoulder, upper_lip_right_shoulder)
    chin_width_proxy = _span(chin_band_points[:, 0])
    semantic_face_extents = _build_semantic_face_extents(
        contour_points=contour_points,
        left_contour_points=left_contour_points,
        right_contour_points=right_contour_points,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        brow_center=brow_center,
        nose_base_center=nose_base_center,
        chin_center=chin_center,
        jaw_left=jaw_left,
        jaw_right=jaw_right,
        cheekbone_left=cheekbone_left,
        cheekbone_right=cheekbone_right,
        detector_bbox=bbox,
    )
    if semantic_face_extents is not None:
        display_face_box = _clip_box_to_image(
            [
                semantic_face_extents["left"],
                semantic_face_extents["top"],
                semantic_face_extents["right"],
                semantic_face_extents["bottom"],
            ],
            image_width=x2,
            image_height=y2,
        )
        face_center_x = float((display_face_box[0] + display_face_box[2]) / 2.0)
    else:
        display_face_box = bbox

    context.update(
        {
            "face_center_x": face_center_x,
            "bbox_top": y1,
            "bbox_bottom": y2,
            "detector_bbox": bbox,
            "semantic_face_extents": semantic_face_extents,
            "display_face_box": display_face_box,
            "left_eye_inner": left_eye_inner,
            "left_eye_outer": left_eye_outer,
            "right_eye_inner": right_eye_inner,
            "right_eye_outer": right_eye_outer,
            "left_eye_top": left_eye_top,
            "left_eye_bottom": left_eye_bottom,
            "right_eye_top": right_eye_top,
            "right_eye_bottom": right_eye_bottom,
            "eye_center": eye_center,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "brow_center": brow_center,
            "nose_mid_left": nose_mid_left,
            "nose_mid_right": nose_mid_right,
            "nose_base_center": nose_base_center,
            "nose_bridge_top": nose_bridge_top,
            "philtrum_top": philtrum_top,
            "lip_midline_center": lip_midline_center,
            "chin_center": chin_center,
            "jaw_left": jaw_left,
            "jaw_right": jaw_right,
            "cheekbone_left": cheekbone_left,
            "cheekbone_right": cheekbone_right,
            "philtrum_width_proxy": philtrum_width_proxy,
            "chin_width_proxy": chin_width_proxy,
            "upper_lip_left_shoulder": upper_lip_left_shoulder,
            "upper_lip_right_shoulder": upper_lip_right_shoulder,
        }
    )

    register("left_eye_center", left_eye_center, group="left_eye", source_type="insightface_106_cluster", source_indices=left_eye_indices.tolist(), description="Mean of the left-eye cluster (33..42).")
    register("right_eye_center", right_eye_center, group="right_eye", source_type="insightface_106_cluster", source_indices=right_eye_indices.tolist(), description="Mean of the right-eye cluster (87..96).")
    register("nose_tip", nose_tip, group="nose", source_type="insightface_106_index", source_indices=[NOSE_TIP_INDEX], description="Nose tip from index 86.", source_rule="canonical_index_86")
    register("nose_bridge_top", nose_bridge_top, group="nose", source_type="insightface_106_index", source_indices=[int(NOSE_BRIDGE_INDICES[0])], description="Top of the canonical nose bridge from index 72.", source_rule="canonical_index_72")
    register("left_mouth", left_mouth, group="mouth", source_type="insightface_106_index", source_indices=[MOUTH_LEFT_CORNER_INDEX], description="Left mouth corner from index 52.")
    register("right_mouth", right_mouth, group="mouth", source_type="insightface_106_index", source_indices=[MOUTH_RIGHT_CORNER_INDEX], description="Right mouth corner from index 61.")
    register("mouth_center", mouth_center, group="mouth", source_type="derived", source_indices=[MOUTH_LEFT_CORNER_INDEX, MOUTH_RIGHT_CORNER_INDEX], description="Midpoint between mouth corners 52 and 61.")
    register("eye_center", eye_center, group="eyes", source_type="derived", source_indices=left_eye_indices.tolist() + right_eye_indices.tolist(), description="Midpoint between left and right eye centers.")
    register("left_eye_outer", left_eye_outer, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_outer_indices, description="Outer corner of the left eye from the left-eye cluster.")
    register("left_eye_inner", left_eye_inner, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_inner_indices, description="Inner corner of the left eye from the left-eye cluster.")
    register("right_eye_inner", right_eye_inner, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_inner_indices, description="Inner corner of the right eye from the right-eye cluster.")
    register("right_eye_outer", right_eye_outer, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_outer_indices, description="Outer corner of the right eye from the right-eye cluster.")
    register("left_eye_top", left_eye_top, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_top_indices, description="Top point of the left-eye cluster.")
    register("left_eye_bottom", left_eye_bottom, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_bottom_indices, description="Bottom point of the left-eye cluster.")
    register("right_eye_top", right_eye_top, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_top_indices, description="Top point of the right-eye cluster.")
    register("right_eye_bottom", right_eye_bottom, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_bottom_indices, description="Bottom point of the right-eye cluster.")
    register("left_brow", left_brow, group="left_brow", source_type="insightface_106_cluster", source_indices=left_brow_indices.tolist(), description="Mean of the left brow cluster (43..51).")
    register("right_brow", right_brow, group="right_brow", source_type="insightface_106_cluster", source_indices=right_brow_indices.tolist(), description="Mean of the right brow cluster (97..105).")
    register("brow_center", brow_center, group="brows", source_type="derived", source_indices=left_brow_indices.tolist() + right_brow_indices.tolist(), description="Midpoint between left and right brow cluster centers.")
    register("alare_left", alare_left, group="nose", source_type="insightface_106_index", source_indices=[ALARE_LEFT_INDEX], description="Left alare from index 77.", source_rule="canonical_index_77")
    register("nose_mid_left", nose_mid_left, group="nose", source_type="insightface_106_index", source_indices=[ALARE_LEFT_INDEX], description="Left nasal width point from alare index 77.", source_rule="canonical_index_77_used_as_nasal_width_anchor")
    register("nose_base_left", nose_base_left, group="nose", source_type="insightface_106_index", source_indices=[NOSE_BASE_LEFT_INDEX], description="Left nose base from index 78.", source_rule="canonical_index_78")
    register("subnasale_left", subnasale_left, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_LEFT_INDEX], description="Left subnasale-adjacent point from index 79.", source_rule="canonical_index_79")
    register("subnasale_center", subnasale_center, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_CENTER_INDEX], description="Subnasale center from index 80.", source_rule="canonical_index_80")
    register("alare_right", alare_right, group="nose", source_type="insightface_106_index", source_indices=[ALARE_RIGHT_INDEX], description="Right alare from index 83.", source_rule="canonical_index_83")
    register("nose_mid_right", nose_mid_right, group="nose", source_type="insightface_106_index", source_indices=[ALARE_RIGHT_INDEX], description="Right nasal width point from alare index 83.", source_rule="canonical_index_83_used_as_nasal_width_anchor")
    register("nose_base_right", nose_base_right, group="nose", source_type="insightface_106_index", source_indices=[NOSE_BASE_RIGHT_INDEX], description="Right nose base from index 84.", source_rule="canonical_index_84")
    register("subnasale_right", subnasale_right, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_RIGHT_INDEX], description="Right subnasale-adjacent point from index 85.", source_rule="canonical_index_85")
    register("nose_base_center", nose_base_center, group="nose", source_type="derived", source_indices=[NOSE_BASE_LEFT_INDEX, NOSE_BASE_RIGHT_INDEX], description="Midpoint between nose-base points 78 and 84.", source_rule="midpoint(nose_base_left,nose_base_right)")
    register("philtrum_top", philtrum_top, group="mouth", source_type="insightface_106_index", source_indices=[UPPER_LIP_CENTER_TOP_INDEX], description="Upper lip center-top used as philtrum-top proxy.")
    register("upper_lip_top", upper_lip_top, group="mouth", source_type="insightface_106_index", source_indices=[UPPER_LIP_CENTER_TOP_INDEX], description="Upper lip center top from index 71.")
    register("lower_lip_bottom", lower_lip_bottom, group="mouth", source_type="insightface_106_index", source_indices=[LOWER_LIP_CENTER_BOTTOM_INDEX], description="Lower lip center bottom from index 53.")
    register("upper_lip_left_shoulder", upper_lip_left_shoulder, group="mouth", source_type="insightface_106_index", source_indices=[63], description="Left shoulder of the upper lip arc used as a philtrum-width proxy anchor.")
    register("upper_lip_right_shoulder", upper_lip_right_shoulder, group="mouth", source_type="insightface_106_index", source_indices=[67], description="Right shoulder of the upper lip arc used as a philtrum-width proxy anchor.")
    register("lip_midline_center", lip_midline_center, group="mouth", source_type="derived", source_indices=[UPPER_LIP_CENTER_TOP_INDEX, LOWER_LIP_CENTER_BOTTOM_INDEX], description="Midpoint between upper lip top and lower lip bottom used to split lip height into upper and lower components.")
    register("chin_center", chin_center, group="contour", source_type="insightface_106_index", source_indices=[0], description="Chin center from contour index 0.")
    register("jaw_left", jaw_left, group="contour", source_type="contour_region_extreme", source_indices=jaw_left_indices, description="Left jaw proxy from lower left contour.")
    register("jaw_right", jaw_right, group="contour", source_type="contour_region_extreme", source_indices=jaw_right_indices, description="Right jaw proxy from lower right contour.")
    register("cheekbone_left", cheekbone_left, group="contour", source_type="contour_region_extreme", source_indices=cheekbone_left_indices, description="Left cheekbone proxy from mid-face left contour.")
    register("cheekbone_right", cheekbone_right, group="contour", source_type="contour_region_extreme", source_indices=cheekbone_right_indices, description="Right cheekbone proxy from mid-face right contour.")

    validation = _validate_semantic_geometry(
        bbox=bbox,
        left_eye=left_eye_center,
        right_eye=right_eye_center,
        nose_tip=nose_tip,
        nose_base_left=nose_base_left,
        nose_base_right=nose_base_right,
        mouth_left=left_mouth,
        mouth_right=right_mouth,
        mouth_center=mouth_center,
        chin_center=chin_center,
        jaw_left=jaw_left,
        jaw_right=jaw_right,
        cheekbone_left=cheekbone_left,
        cheekbone_right=cheekbone_right,
        semantic_metadata=semantic_meta,
    )
    status = "validated"
    if not bool(validation.get("passed")):
        status = "unreliable"
    elif validation.get("warning_checks"):
        status = "warning"
    for meta in semantic_meta.values():
        meta["validation_status"] = status
    context["semantic_metadata"] = semantic_meta
    context["semantic_validation"] = validation
    return context


def _derive_dense_context(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, np.ndarray | float | None]:
    context: dict[str, np.ndarray | float | None] = {}
    if not landmarks or not bbox or len(landmarks) < 5 or len(bbox) != 4:
        return context

    points5 = np.asarray(landmarks[:5], dtype=np.float64)
    left_eye, right_eye, nose_tip, left_mouth, right_mouth = points5
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0

    dense = np.asarray(landmarks_dense, dtype=np.float64) if landmarks_dense else None
    context.update(
        {
            "left_eye_center": left_eye,
            "right_eye_center": right_eye,
            "nose_tip": nose_tip,
            "left_mouth": left_mouth,
            "right_mouth": right_mouth,
            "eye_center": eye_center,
            "mouth_center": mouth_center,
            "face_center_x": face_center_x,
            "bbox_top": y1,
            "bbox_bottom": y2,
        }
    )

    if dense is None or len(dense) == 0:
        return context

    if has_106_landmarks(dense):
        return _derive_dense_context_from_insightface_106(
            landmarks=landmarks,
            landmarks_dense=landmarks_dense or [],
            bbox=bbox,
        )

    mouth_geometry = _derive_mouth_geometry_from_dense(
        dense=dense,
        bbox=bbox,
        nose_tip=nose_tip,
        fallback_left=left_mouth,
        fallback_right=right_mouth,
    )
    left_mouth = mouth_geometry["left_mouth"]
    right_mouth = mouth_geometry["right_mouth"]
    mouth_center = mouth_geometry["mouth_center"]
    context.update(
        {
            "left_mouth": left_mouth,
            "right_mouth": right_mouth,
            "mouth_center": mouth_center,
        }
    )

    left_eye_region = _region_points(
        dense,
        x_min=float(x1),
        x_max=float(face_center_x),
        y_min=float(left_eye[1] - 0.12 * face_height),
        y_max=float(left_eye[1] + 0.08 * face_height),
    )
    right_eye_region = _region_points(
        dense,
        x_min=float(face_center_x),
        x_max=float(x2),
        y_min=float(right_eye[1] - 0.12 * face_height),
        y_max=float(right_eye[1] + 0.08 * face_height),
    )

    left_eye_outer = _extreme_point(left_eye_region, axis=0, mode="min")
    left_eye_inner = _extreme_point(left_eye_region, axis=0, mode="max")
    right_eye_inner = _extreme_point(right_eye_region, axis=0, mode="min")
    right_eye_outer = _extreme_point(right_eye_region, axis=0, mode="max")
    left_eye_top = _extreme_point(left_eye_region, axis=1, mode="min")
    left_eye_bottom = _extreme_point(left_eye_region, axis=1, mode="max")
    right_eye_top = _extreme_point(right_eye_region, axis=1, mode="min")
    right_eye_bottom = _extreme_point(right_eye_region, axis=1, mode="max")

    left_brow_region = _region_points(
        dense,
        x_min=float(left_eye[0] - 0.18 * face_width),
        x_max=float(left_eye[0] + 0.10 * face_width),
        y_min=float(left_eye[1] - 0.24 * face_height),
        y_max=float(left_eye[1] - 0.02 * face_height),
    )
    right_brow_region = _region_points(
        dense,
        x_min=float(right_eye[0] - 0.10 * face_width),
        x_max=float(right_eye[0] + 0.18 * face_width),
        y_min=float(right_eye[1] - 0.24 * face_height),
        y_max=float(right_eye[1] - 0.02 * face_height),
    )
    left_brow = _cluster_center(left_brow_region)
    right_brow = _cluster_center(right_brow_region)

    nose_mid_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.16 * face_width),
        x_max=float(face_center_x + 0.16 * face_width),
        y_min=float(nose_tip[1] - 0.18 * face_height),
        y_max=float(nose_tip[1] - 0.02 * face_height),
    )
    nose_base_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.18 * face_width),
        x_max=float(face_center_x + 0.18 * face_width),
        y_min=float(nose_tip[1] - 0.05 * face_height),
        y_max=float(nose_tip[1] + 0.14 * face_height),
    )
    nose_mid_left = _extreme_point(nose_mid_region, axis=0, mode="min")
    nose_mid_right = _extreme_point(nose_mid_region, axis=0, mode="max")
    nose_base_left = _extreme_point(nose_base_region, axis=0, mode="min")
    nose_base_right = _extreme_point(nose_base_region, axis=0, mode="max")
    nose_base_left_source_indices: list[int] = []
    nose_base_right_source_indices: list[int] = []
    if has_106_landmarks(dense):
        mapped_nose_base_left = _safe_index_point(dense, int(NOSE_BASE_LEFT_INDEX))
        mapped_nose_base_right = _safe_index_point(dense, int(NOSE_BASE_RIGHT_INDEX))
        if mapped_nose_base_left is not None:
            if (
                abs(float(mapped_nose_base_left[0] - face_center_x)) <= 0.28 * face_width
                and float(mapped_nose_base_left[1]) >= float(nose_tip[1]) - 0.10 * face_height
                and float(mapped_nose_base_left[1]) <= float(nose_tip[1]) + 0.22 * face_height
            ):
                nose_base_left = mapped_nose_base_left
                nose_base_left_source_indices = [int(NOSE_BASE_LEFT_INDEX)]
        if mapped_nose_base_right is not None:
            if (
                abs(float(mapped_nose_base_right[0] - face_center_x)) <= 0.28 * face_width
                and float(mapped_nose_base_right[1]) >= float(nose_tip[1]) - 0.10 * face_height
                and float(mapped_nose_base_right[1]) <= float(nose_tip[1]) + 0.22 * face_height
            ):
                nose_base_right = mapped_nose_base_right
                nose_base_right_source_indices = [int(NOSE_BASE_RIGHT_INDEX)]
    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    mouth_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.26 * face_width),
        x_max=float(face_center_x + 0.26 * face_width),
        y_min=float(max(nose_tip[1] + 0.05 * face_height, y1 + 0.56 * face_height)),
        y_max=float(y1 + 0.84 * face_height),
    )
    central_lip_region = _region_points(
        mouth_region,
        x_min=float(face_center_x - 0.14 * face_width),
        x_max=float(face_center_x + 0.14 * face_width),
    )
    upper_lip_top = mouth_geometry.get("upper_lip_top")
    if upper_lip_top is None:
        upper_lip_top = _closest_point_to_x(
            central_lip_region,
            face_center_x,
            mode_y="min",
        )
    lower_lip_bottom = mouth_geometry.get("lower_lip_bottom")
    if lower_lip_bottom is None:
        lower_lip_bottom = _closest_point_to_x(
            central_lip_region,
            face_center_x,
            mode_y="max",
        )
    philtrum_top_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.10 * face_width),
        x_max=float(face_center_x + 0.10 * face_width),
        y_min=float((nose_base_center[1] if nose_base_center is not None else nose_tip[1]) - 0.02 * face_height),
        y_max=float(mouth_center[1]),
    )
    philtrum_top = _closest_point_to_x(philtrum_top_region, face_center_x, mode_y="min")
    lip_midline_center = _point_midpoint(upper_lip_top, lower_lip_bottom)

    chin_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.15 * face_width),
        x_max=float(face_center_x + 0.15 * face_width),
        y_min=float(mouth_center[1] + 0.08 * face_height),
        y_max=float(y2),
    )
    chin_center = _closest_point_to_x(chin_region, face_center_x, mode_y="max")

    chin_band = _region_points(
        dense,
        x_min=float(face_center_x - 0.20 * face_width),
        x_max=float(face_center_x + 0.20 * face_width),
        y_min=float((chin_center[1] if chin_center is not None else mouth_center[1]) - 0.06 * face_height),
        y_max=float((chin_center[1] if chin_center is not None else y2) + 0.01 * face_height),
    )
    jaw_region_y_min = float(max(y1 + 0.74 * face_height, mouth_center[1] + 0.02 * face_height))
    jaw_source = dense[np.asarray(CONTOUR_INDICES, dtype=np.int32)] if has_106_landmarks(dense) else dense
    jaw_region = _region_points(
        jaw_source,
        y_min=jaw_region_y_min,
        y_max=float(y1 + 0.98 * face_height),
    )
    cheekbone_region = _region_points(
        dense,
        y_min=float(y1 + 0.35 * face_height),
        y_max=float(y1 + 0.65 * face_height),
    )

    context.update(
        {
            "left_eye_inner": left_eye_inner,
            "left_eye_outer": left_eye_outer,
            "right_eye_inner": right_eye_inner,
            "right_eye_outer": right_eye_outer,
            "left_eye_top": left_eye_top,
            "left_eye_bottom": left_eye_bottom,
            "right_eye_top": right_eye_top,
            "right_eye_bottom": right_eye_bottom,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "nose_mid_left": nose_mid_left,
            "nose_mid_right": nose_mid_right,
            "nose_base_left": nose_base_left,
            "nose_base_right": nose_base_right,
            "nose_base_center": nose_base_center,
            "nose_base_left_source_indices": nose_base_left_source_indices,
            "nose_base_right_source_indices": nose_base_right_source_indices,
            "upper_lip_top": upper_lip_top,
            "lower_lip_bottom": lower_lip_bottom,
            "philtrum_top": philtrum_top,
            "lip_midline_center": lip_midline_center,
            "chin_center": chin_center,
            "jaw_left": _extreme_point(jaw_region, axis=0, mode="min"),
            "jaw_right": _extreme_point(jaw_region, axis=0, mode="max"),
            "cheekbone_left": _extreme_point(cheekbone_region, axis=0, mode="min"),
            "cheekbone_right": _extreme_point(cheekbone_region, axis=0, mode="max"),
            "philtrum_width_proxy": _span(_region_points(mouth_region, y_min=float((upper_lip_top[1] if upper_lip_top is not None else mouth_center[1]) - 0.02 * face_height), y_max=float((upper_lip_top[1] if upper_lip_top is not None else mouth_center[1]) + 0.02 * face_height))[:, 0]),
            "chin_width_proxy": _span(chin_band[:, 0]),
        }
    )
    validation = _validate_semantic_geometry(
        bbox=bbox,
        left_eye=left_eye,
        right_eye=right_eye,
        nose_tip=nose_tip,
        nose_base_left=nose_base_left,
        nose_base_right=nose_base_right,
        mouth_left=left_mouth,
        mouth_right=right_mouth,
        mouth_center=mouth_center,
        chin_center=chin_center,
        jaw_left=context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None,
        jaw_right=context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None,
        cheekbone_left=context.get("cheekbone_left") if isinstance(context.get("cheekbone_left"), np.ndarray) else None,
        cheekbone_right=context.get("cheekbone_right") if isinstance(context.get("cheekbone_right"), np.ndarray) else None,
        semantic_metadata=context.get("semantic_metadata") if isinstance(context.get("semantic_metadata"), dict) else None,
    )
    context["semantic_validation"] = validation
    return context


def derive_dense_visual_context(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, np.ndarray | float | None]:
    return _derive_dense_context(
        landmarks=landmarks,
        landmarks_dense=landmarks_dense,
        bbox=bbox,
    )


def derive_semantic_landmark_bundle(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, Any]:
    if not landmarks or len(landmarks) < 5:
        return {
            "raw_landmarks": [],
            "raw_landmarks_debug": [],
            "semantic_points": {},
            "semantic_points_debug": [],
        }

    def _optional_point(name: str) -> np.ndarray | None:
        value = context.get(name)
        return value if isinstance(value, np.ndarray) else None

    def _optional_source_indices(name: str) -> list[int]:
        value = context.get(f"{name}_source_indices")
        if not isinstance(value, list):
            return []
        return [int(item) for item in value if isinstance(item, (int, np.integer))]

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    left_eye = _optional_point("left_eye_center")
    right_eye = _optional_point("right_eye_center")
    nose_tip = _optional_point("nose_tip")
    left_mouth = _optional_point("left_mouth")
    right_mouth = _optional_point("right_mouth")
    mouth_center = _optional_point("mouth_center")

    semantic_meta_by_name = (
        context.get("semantic_metadata")
        if isinstance(context.get("semantic_metadata"), dict)
        else {}
    )
    semantic_meta_aliases = {
        "mouth_left": "left_mouth",
        "mouth_right": "right_mouth",
        "brow_left": "left_brow",
        "brow_right": "right_brow",
    }

    semantic_specs: list[tuple[str, np.ndarray | None, str, str, list[int], str]] = [
        ("left_eye_center", left_eye, "eyes", "anchor", [0], "Left eye center used in morphometric measurements."),
        ("right_eye_center", right_eye, "eyes", "anchor", [1], "Right eye center used in morphometric measurements."),
        ("nose_tip", nose_tip, "nose", "anchor", [2], "Nasal tip used in morphometric measurements."),
        ("nose_bridge_top", _optional_point("nose_bridge_top"), "nose", "derived", _optional_source_indices("nose_bridge_top"), "Top of the canonical nose bridge used in the corrected nose-length proxy."),
        ("left_eye_outer", _optional_point("left_eye_outer"), "eyes", "derived", [], "Left eye outer corner constrained to the left-eye cluster."),
        ("left_eye_inner", _optional_point("left_eye_inner"), "eyes", "derived", [], "Left eye inner corner constrained to the left-eye cluster."),
        ("right_eye_inner", _optional_point("right_eye_inner"), "eyes", "derived", [], "Right eye inner corner constrained to the right-eye cluster."),
        ("right_eye_outer", _optional_point("right_eye_outer"), "eyes", "derived", [], "Right eye outer corner constrained to the right-eye cluster."),
        ("left_eye_top", _optional_point("left_eye_top"), "eyes", "derived", [], "Top point of the left-eye cluster."),
        ("left_eye_bottom", _optional_point("left_eye_bottom"), "eyes", "derived", [], "Bottom point of the left-eye cluster."),
        ("right_eye_top", _optional_point("right_eye_top"), "eyes", "derived", [], "Top point of the right-eye cluster."),
        ("right_eye_bottom", _optional_point("right_eye_bottom"), "eyes", "derived", [], "Bottom point of the right-eye cluster."),
        ("mouth_left", left_mouth, "mouth", "derived", [], "Left mouth corner used in morphometric measurements."),
        ("mouth_right", right_mouth, "mouth", "derived", [], "Right mouth corner used in morphometric measurements."),
        ("mouth_center", mouth_center, "mouth", "derived", [], "Midpoint between the left and right mouth corners."),
        ("eye_center", _optional_point("eye_center"), "eyes", "derived", [0, 1], "Midpoint between the eye centers."),
        ("brow_left", _optional_point("left_brow"), "brows", "derived", [], "Estimated left brow point from dense brow landmarks."),
        ("brow_right", _optional_point("right_brow"), "brows", "derived", [], "Estimated right brow point from dense brow landmarks."),
        ("brow_center", _optional_point("brow_center"), "brows", "derived", [], "Midpoint between left and right brow points."),
        ("nose_mid_left", _optional_point("nose_mid_left"), "nose", "derived", [], "Left point on the mid-nose width band."),
        ("nose_mid_right", _optional_point("nose_mid_right"), "nose", "derived", [], "Right point on the mid-nose width band."),
        (
            "nose_base_left",
            _optional_point("nose_base_left"),
            "nose",
            "derived",
            _optional_source_indices("nose_base_left"),
            "Left nose-base point from dense landmarks.",
        ),
        (
            "nose_base_right",
            _optional_point("nose_base_right"),
            "nose",
            "derived",
            _optional_source_indices("nose_base_right"),
            "Right nose-base point from dense landmarks.",
        ),
        ("alare_left", _optional_point("alare_left"), "nose", "derived", _optional_source_indices("alare_left"), "Left alare point from InsightFace 106 nose region."),
        ("subnasale_left", _optional_point("subnasale_left"), "nose", "derived", _optional_source_indices("subnasale_left"), "Left subnasale-adjacent point from InsightFace 106 nose region."),
        ("subnasale_center", _optional_point("subnasale_center"), "nose", "derived", _optional_source_indices("subnasale_center"), "Subnasale center from InsightFace 106 nose region."),
        ("subnasale_right", _optional_point("subnasale_right"), "nose", "derived", _optional_source_indices("subnasale_right"), "Right subnasale-adjacent point from InsightFace 106 nose region."),
        ("alare_right", _optional_point("alare_right"), "nose", "derived", _optional_source_indices("alare_right"), "Right alare point from InsightFace 106 nose region."),
        ("nose_base_center", _optional_point("nose_base_center"), "nose", "derived", [], "Midpoint of the nose base."),
        ("philtrum_top", _optional_point("philtrum_top"), "mouth", "derived", [], "Upper philtrum point near the face midline."),
        ("upper_lip_top", _optional_point("upper_lip_top"), "mouth", "derived", [], "Upper lip top near the mouth midline."),
        ("lower_lip_bottom", _optional_point("lower_lip_bottom"), "mouth", "derived", [], "Lower lip bottom near the mouth midline."),
        ("upper_lip_left_shoulder", _optional_point("upper_lip_left_shoulder"), "mouth", "derived", _optional_source_indices("upper_lip_left_shoulder"), "Left upper-lip shoulder used in the philtrum-width proxy."),
        ("upper_lip_right_shoulder", _optional_point("upper_lip_right_shoulder"), "mouth", "derived", _optional_source_indices("upper_lip_right_shoulder"), "Right upper-lip shoulder used in the philtrum-width proxy."),
        ("lip_midline_center", _optional_point("lip_midline_center"), "mouth", "derived", [], "Midpoint between upper and lower lip central anchors."),
        ("chin_center", _optional_point("chin_center"), "chin_jaw", "derived", [], "Lowest central chin point."),
        ("jaw_left", _optional_point("jaw_left"), "chin_jaw", "derived", [], "Left jaw proxy point from lower contour landmarks."),
        ("jaw_right", _optional_point("jaw_right"), "chin_jaw", "derived", [], "Right jaw proxy point from lower contour landmarks."),
        ("cheekbone_left", _optional_point("cheekbone_left"), "chin_jaw", "derived", [], "Left cheekbone proxy point from mid-face contour landmarks."),
        ("cheekbone_right", _optional_point("cheekbone_right"), "chin_jaw", "derived", [], "Right cheekbone proxy point from mid-face contour landmarks."),
    ]

    semantic_points: dict[str, dict[str, Any]] = {}
    semantic_points_debug: list[dict[str, Any]] = []
    for name, point, group, source_type, source_indices, description in semantic_specs:
        if point is None:
            continue
        debug_entry = semantic_meta_by_name.get(name)
        if debug_entry is None:
            debug_entry = semantic_meta_by_name.get(semantic_meta_aliases.get(name, ""))
        if debug_entry is None:
            debug_entry = _semantic_point_debug_entry(
                point_name=name,
                point=point,
                group=group,
                source_type=source_type,
                source_indices=source_indices,
                description=description,
                source_region=_region_for_source_indices(source_indices),
                validation_status=(
                    "validated"
                    if not isinstance(context.get("semantic_validation"), dict)
                    or bool(context.get("semantic_validation", {}).get("passed", False))
                    else "unreliable"
                ),
            )
        else:
            debug_entry = dict(debug_entry)
            debug_entry["point_name"] = name
        semantic_points[name] = _semantic_point_payload(
            point_name=name,
            point=point,
            source_type=str(debug_entry.get("source_type", source_type)),
            source_indices=list(debug_entry.get("source_indices", source_indices)),
            source_region=debug_entry.get("source_region"),
            validation_status=str(debug_entry.get("validation_status", "validated")),
            source_rule=debug_entry.get("source_rule"),
        ) or {"x": float(point[0]), "y": float(point[1])}
        if debug_entry is not None:
            semantic_points_debug.append(debug_entry)

    dense = np.asarray(landmarks_dense, dtype=np.float64) if landmarks_dense else None
    raw_source = dense if dense is not None and len(dense) > 0 else np.asarray(landmarks[:5], dtype=np.float64)
    raw_group = "dense_landmarks" if dense is not None and len(dense) > 0 else "anchors"
    def _raw_group_for_index(index: int) -> str:
        if raw_group != "dense_landmarks":
            return raw_group
        return region_name_for_index(index)

    raw_landmarks = [
        {
            "index": int(index),
            "x": float(point[0]),
            "y": float(point[1]),
            "group": _raw_group_for_index(int(index)),
            "name": None,
            "source_type": "raw",
            "source_indices": [int(index)],
            "description": (
                "Raw landmark emitted by the dense InsightFace landmark_2d_106 model."
                if raw_group == "dense_landmarks"
                else "Raw five-point anchor landmark."
            ),
        }
        for index, point in enumerate(raw_source.tolist())
    ]
    raw_landmarks_debug = [
        {
            "index": int(index),
            "x": float(point[0]),
            "y": float(point[1]),
            "group": _raw_group_for_index(int(index)),
            "name": None,
            "source_type": "raw",
            "source_indices": [int(index)],
            "description": (
                "Raw landmark emitted by the dense InsightFace landmark_2d_106 model."
                if raw_group == "dense_landmarks"
                else "Raw five-point anchor landmark."
            ),
        }
        for index, point in enumerate(raw_source.tolist())
    ]
    semantic_validation = context.get("semantic_validation")
    return {
        "raw_landmarks": raw_landmarks,
        "raw_landmarks_debug": raw_landmarks_debug,
        "semantic_points": semantic_points,
        "semantic_points_debug": semantic_points_debug,
        "semantic_validation": semantic_validation,
    }


def extract_raw_measurements(
    landmarks: list[list[float]] | None,
    bbox: list[float] | None,
    landmarks_dense: list[list[float]] | None = None,
    mediapipe_landmarks: list[list[float]] | None = None,
) -> dict[str, float | None]:
    if not landmarks or not bbox or len(landmarks) < 5 or len(bbox) != 4:
        result = empty_feature_dict(RAW_MEASUREMENT_COLUMNS)
        result.update(_mediapipe_raw_measurements(mediapipe_landmarks))
        return result

    points = np.asarray(landmarks, dtype=np.float64)
    left_eye, right_eye, nose = points[:3]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    detector_face_width = x2 - x1
    detector_face_height = y2 - y1
    face_width = detector_face_width
    face_height = detector_face_height
    if face_width <= 0.0 or face_height <= 0.0:
        return empty_feature_dict(RAW_MEASUREMENT_COLUMNS)

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    left_eye = context.get("left_eye_center") if isinstance(context.get("left_eye_center"), np.ndarray) else left_eye
    right_eye = context.get("right_eye_center") if isinstance(context.get("right_eye_center"), np.ndarray) else right_eye
    nose = context.get("nose_tip") if isinstance(context.get("nose_tip"), np.ndarray) else nose
    left_mouth = context.get("left_mouth") if isinstance(context.get("left_mouth"), np.ndarray) else points[3]
    right_mouth = context.get("right_mouth") if isinstance(context.get("right_mouth"), np.ndarray) else points[4]
    eye_center = context.get("eye_center") if isinstance(context.get("eye_center"), np.ndarray) else (left_eye + right_eye) / 2.0
    mouth_center = context.get("mouth_center") if isinstance(context.get("mouth_center"), np.ndarray) else (left_mouth + right_mouth) / 2.0
    semantic_face_extents = context.get("semantic_face_extents") if isinstance(context.get("semantic_face_extents"), dict) else None
    display_face_box = context.get("display_face_box") if isinstance(context.get("display_face_box"), list) else bbox
    if semantic_face_extents is not None:
        x1 = float(semantic_face_extents["left"])
        y1 = float(semantic_face_extents["top"])
        x2 = float(semantic_face_extents["right"])
        y2 = float(semantic_face_extents["bottom"])
        face_width = float(semantic_face_extents["width"])
        face_height = float(semantic_face_extents["height"])
    face_center_x = float(context.get("face_center_x")) if context.get("face_center_x") is not None else (x1 + x2) / 2.0

    left_brow = context.get("left_brow")
    right_brow = context.get("right_brow")
    brow_center = _point_midpoint(
        left_brow if isinstance(left_brow, np.ndarray) else None,
        right_brow if isinstance(right_brow, np.ndarray) else None,
    )
    nose_base_center = context.get("nose_base_center") if isinstance(context.get("nose_base_center"), np.ndarray) else None
    upper_lip_top = context.get("upper_lip_top") if isinstance(context.get("upper_lip_top"), np.ndarray) else None
    lower_lip_bottom = context.get("lower_lip_bottom") if isinstance(context.get("lower_lip_bottom"), np.ndarray) else None
    chin_center = context.get("chin_center") if isinstance(context.get("chin_center"), np.ndarray) else None
    left_eye_inner = context.get("left_eye_inner") if isinstance(context.get("left_eye_inner"), np.ndarray) else None
    left_eye_outer = context.get("left_eye_outer") if isinstance(context.get("left_eye_outer"), np.ndarray) else None
    right_eye_inner = context.get("right_eye_inner") if isinstance(context.get("right_eye_inner"), np.ndarray) else None
    right_eye_outer = context.get("right_eye_outer") if isinstance(context.get("right_eye_outer"), np.ndarray) else None
    left_eye_top = context.get("left_eye_top") if isinstance(context.get("left_eye_top"), np.ndarray) else None
    left_eye_bottom = context.get("left_eye_bottom") if isinstance(context.get("left_eye_bottom"), np.ndarray) else None
    right_eye_top = context.get("right_eye_top") if isinstance(context.get("right_eye_top"), np.ndarray) else None
    right_eye_bottom = context.get("right_eye_bottom") if isinstance(context.get("right_eye_bottom"), np.ndarray) else None
    nose_mid_left = context.get("nose_mid_left") if isinstance(context.get("nose_mid_left"), np.ndarray) else None
    nose_mid_right = context.get("nose_mid_right") if isinstance(context.get("nose_mid_right"), np.ndarray) else None
    nose_base_left = context.get("nose_base_left") if isinstance(context.get("nose_base_left"), np.ndarray) else None
    nose_base_right = context.get("nose_base_right") if isinstance(context.get("nose_base_right"), np.ndarray) else None
    nose_bridge_top = context.get("nose_bridge_top") if isinstance(context.get("nose_bridge_top"), np.ndarray) else None
    subnasale_center = context.get("subnasale_center") if isinstance(context.get("subnasale_center"), np.ndarray) else None
    philtrum_top = context.get("philtrum_top") if isinstance(context.get("philtrum_top"), np.ndarray) else None
    lip_midline_center = context.get("lip_midline_center") if isinstance(context.get("lip_midline_center"), np.ndarray) else mouth_center
    jaw_left = context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None
    jaw_right = context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None
    cheekbone_left = context.get("cheekbone_left") if isinstance(context.get("cheekbone_left"), np.ndarray) else None
    cheekbone_right = context.get("cheekbone_right") if isinstance(context.get("cheekbone_right"), np.ndarray) else None

    left_eye_width = _distance(left_eye_outer, left_eye_inner)
    right_eye_width = _distance(right_eye_inner, right_eye_outer)
    left_eye_height = _distance(left_eye_top, left_eye_bottom)
    right_eye_height = _distance(right_eye_top, right_eye_bottom)
    intercanthal_distance = _distance(left_eye_inner, right_eye_inner)
    outercanthal_distance = _distance(left_eye_outer, right_eye_outer)
    nose_width = _distance(nose_mid_left, nose_mid_right)
    nose_base_width = _distance(nose_base_left, nose_base_right)
    mouth_center_to_chin = abs(float(chin_center[1] - mouth_center[1])) if chin_center is not None else y2 - float(mouth_center[1])
    mouth_center_midline_offset = abs(float(mouth_center[0]) - face_center_x)
    nose_tip_midline_offset = abs(float(nose[0]) - face_center_x)
    chin_center_midline_offset = abs(float(chin_center[0]) - face_center_x) if chin_center is not None else None
    philtrum_midline_offset_proxy = abs(float(philtrum_top[0]) - face_center_x) if philtrum_top is not None else None

    result = {
        "raw_face_width_px": face_width,
        "raw_face_height_px": face_height,
        "raw_eye_distance_px": _distance(left_eye, right_eye),
        "raw_mouth_width_px": _distance(left_mouth, right_mouth),
        "raw_eye_to_nose_px": _distance(eye_center, nose),
        "raw_nose_to_mouth_px": _distance(nose, mouth_center),
        "raw_eye_to_mouth_px": _distance(eye_center, mouth_center),
        "raw_left_eye_to_nose_px": _distance(left_eye, nose),
        "raw_right_eye_to_nose_px": _distance(right_eye, nose),
        "raw_left_mouth_to_nose_px": _distance(left_mouth, nose),
        "raw_right_mouth_to_nose_px": _distance(right_mouth, nose),
        "raw_upper_face_height_px": float(nose[1]) - y1,
        "raw_lower_face_height_px": mouth_center_to_chin,
        "raw_forehead_height_proxy_px": float(eye_center[1]) - y1,
        "raw_jaw_height_proxy_px": mouth_center_to_chin,
        "raw_nose_center_offset_px": nose_tip_midline_offset,
        "raw_mouth_center_offset_px": mouth_center_midline_offset,
        "brow_to_eye_center_height": abs(float(eye_center[1] - brow_center[1])) if brow_center is not None else None,
        "eye_center_to_nose_base_height": abs(float(nose_base_center[1] - eye_center[1])) if nose_base_center is not None else None,
        "nose_base_to_mouth_center_height": abs(float(mouth_center[1] - nose_base_center[1])) if nose_base_center is not None else None,
        "mouth_center_to_chin_height": mouth_center_to_chin,
        "brow_to_nose_base_height": abs(float(nose_base_center[1] - brow_center[1])) if brow_center is not None and nose_base_center is not None else None,
        "nose_base_to_chin_height": abs(float(chin_center[1] - nose_base_center[1])) if chin_center is not None and nose_base_center is not None else None,
        "eye_center_to_chin_height": abs(float(chin_center[1] - eye_center[1])) if chin_center is not None else None,
        "forehead_proxy_to_brow_height": abs(float(brow_center[1] - y1)) if brow_center is not None else None,
        "upper_third_proxy": abs(float(nose_base_center[1] - brow_center[1])) if brow_center is not None and nose_base_center is not None else None,
        "middle_third_proxy": abs(float(nose_base_center[1] - eye_center[1])) if nose_base_center is not None else None,
        "lower_third_proxy": abs(float(chin_center[1] - nose_base_center[1])) if chin_center is not None and nose_base_center is not None else None,
        "intercanthal_distance": intercanthal_distance,
        "outercanthal_distance": outercanthal_distance,
        "left_eye_width": left_eye_width,
        "right_eye_width": right_eye_width,
        "mean_eye_width": _mean([left_eye_width, right_eye_width]),
        "left_eye_height": left_eye_height,
        "right_eye_height": right_eye_height,
        "mean_eye_height": _mean([left_eye_height, right_eye_height]),
        "nose_width": nose_width,
        "nose_base_width": nose_base_width,
        "philtrum_width_proxy": _to_float(context.get("philtrum_width_proxy")),
        "chin_width_proxy": _to_float(context.get("chin_width_proxy")),
        "jaw_width_proxy": _distance(jaw_left, jaw_right),
        "cheekbone_width_proxy": _distance(cheekbone_left, cheekbone_right),
        "left_right_eye_y_diff": abs(float(left_eye[1] - right_eye[1])),
        "left_right_eye_width_diff": _abs_diff(left_eye_width, right_eye_width),
        "left_right_eye_height_diff": _abs_diff(left_eye_height, right_eye_height),
        "left_right_brow_height_diff": _abs_diff(
            abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
            abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        ),
        "left_right_mouth_corner_y_diff": abs(float(left_mouth[1] - right_mouth[1])),
        "left_right_mouth_to_nose_diff": _abs_diff(_distance(left_mouth, nose), _distance(right_mouth, nose)),
        "left_right_eye_to_nose_diff": _abs_diff(_distance(left_eye, nose), _distance(right_eye, nose)),
        "nose_tip_midline_offset": nose_tip_midline_offset,
        "mouth_center_midline_offset": mouth_center_midline_offset,
        "chin_center_midline_offset": chin_center_midline_offset,
        "philtrum_midline_offset_proxy": philtrum_midline_offset_proxy,
        "left_brow_to_eye_distance": abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
        "right_brow_to_eye_distance": abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        "mean_brow_to_eye_distance": _mean([
            abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
            abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        ]),
        "eye_area_proxy_left": (left_eye_width * left_eye_height) if left_eye_width is not None and left_eye_height is not None else None,
        "eye_area_proxy_right": (right_eye_width * right_eye_height) if right_eye_width is not None and right_eye_height is not None else None,
        "eye_area_proxy_mean": _mean([
            (left_eye_width * left_eye_height) if left_eye_width is not None and left_eye_height is not None else None,
            (right_eye_width * right_eye_height) if right_eye_width is not None and right_eye_height is not None else None,
        ]),
        "nose_length": _distance(nose_bridge_top, subnasale_center),
        "nose_tip_to_upper_lip_distance": _distance(nose, upper_lip_top),
        "philtrum_height": abs(float(upper_lip_top[1] - subnasale_center[1])) if upper_lip_top is not None and subnasale_center is not None else None,
        "upper_lip_height": abs(float(lip_midline_center[1] - upper_lip_top[1])) if upper_lip_top is not None and lip_midline_center is not None else None,
        "lower_lip_height": abs(float(lower_lip_bottom[1] - lip_midline_center[1])) if lower_lip_bottom is not None and lip_midline_center is not None else None,
        "chin_height": abs(float(chin_center[1] - lower_lip_bottom[1])) if chin_center is not None and lower_lip_bottom is not None else None,
    }
    result.update(_mediapipe_raw_measurements(mediapipe_landmarks))
    nose_length = result.get("nose_length")
    if nose_length is not None and face_height > 0.0:
        nose_length_ratio = float(nose_length) / float(face_height)
        if nose_length_ratio < 0.05:
            result["nose_length"] = None
    validation = context.get("semantic_validation")
    if isinstance(validation, dict):
        unreliable_features = {
            str(name)
            for name in validation.get("unreliable_features", [])
            if isinstance(name, str)
        }
        for feature_name in unreliable_features:
            if feature_name in result:
                result[feature_name] = None
    return result


def extract_normalized_measurements(
    raw_measurements: dict[str, float | None],
) -> dict[str, float | None]:
    face_width = raw_measurements.get("raw_face_width_px")
    face_height = raw_measurements.get("raw_face_height_px")
    mouth_width = raw_measurements.get("raw_mouth_width_px")
    mean_eye_width = raw_measurements.get("mean_eye_width")
    mean_eye_height = raw_measurements.get("mean_eye_height")
    upper = raw_measurements.get("upper_third_proxy")
    middle = raw_measurements.get("middle_third_proxy")
    lower = raw_measurements.get("lower_third_proxy")
    thirds = [value for value in (upper, middle, lower) if value is not None]
    thirds_balance_index = None
    if len(thirds) == 3 and max(thirds) > 0:
        normalized = np.asarray(thirds, dtype=np.float64) / float(sum(thirds))
        thirds_balance_index = float(max(0.0, 1.0 - min(float(np.std(normalized, ddof=0)) * 3.0, 1.0)))

    result = {
        "ratio_face_aspect": _safe_divide(face_width, face_height),
        "ratio_eye_distance_to_face_width": _safe_divide(raw_measurements.get("raw_eye_distance_px"), face_width),
        "ratio_mouth_width_to_face_width": _safe_divide(mouth_width, face_width),
        "ratio_eye_to_nose_to_face_height": _safe_divide(raw_measurements.get("raw_eye_to_nose_px"), face_height),
        "ratio_nose_to_mouth_to_face_height": _safe_divide(raw_measurements.get("raw_nose_to_mouth_px"), face_height),
        "ratio_eye_to_mouth_to_face_height": _safe_divide(raw_measurements.get("raw_eye_to_mouth_px"), face_height),
        "ratio_eye_distance_to_mouth_width": _safe_divide(raw_measurements.get("raw_eye_distance_px"), mouth_width),
        "ratio_upper_face_height_to_face_height": _safe_divide(raw_measurements.get("raw_upper_face_height_px"), face_height),
        "ratio_lower_face_height_to_face_height": _safe_divide(raw_measurements.get("raw_lower_face_height_px"), face_height),
        "ratio_forehead_height_proxy_to_face_height": _safe_divide(raw_measurements.get("raw_forehead_height_proxy_px"), face_height),
        "ratio_jaw_height_proxy_to_face_height": _safe_divide(raw_measurements.get("raw_jaw_height_proxy_px"), face_height),
        "ratio_nose_center_offset_to_face_width": _safe_divide(raw_measurements.get("raw_nose_center_offset_px"), face_width),
        "ratio_mouth_center_offset_to_face_width": _safe_divide(raw_measurements.get("raw_mouth_center_offset_px"), face_width),
        "ratio_upper_third_to_middle_third": _safe_divide(upper, middle),
        "ratio_middle_third_to_lower_third": _safe_divide(middle, lower),
        "ratio_nose_base_to_mouth_to_lower_third": _safe_divide(raw_measurements.get("nose_base_to_mouth_center_height"), lower),
        "ratio_mouth_to_chin_to_lower_third": _safe_divide(raw_measurements.get("mouth_center_to_chin_height"), lower),
        "thirds_balance_index": thirds_balance_index,
        "ratio_intercanthal_to_eye_width": _safe_divide(raw_measurements.get("intercanthal_distance"), mean_eye_width),
        "ratio_outercanthal_to_face_width": _safe_divide(raw_measurements.get("outercanthal_distance"), face_width),
        "ratio_nose_width_to_face_width": _safe_divide(raw_measurements.get("nose_width"), face_width),
        "ratio_mouth_width_to_nose_width": _safe_divide(mouth_width, raw_measurements.get("nose_width")),
        "ratio_chin_width_to_jaw_width": _safe_divide(raw_measurements.get("chin_width_proxy"), raw_measurements.get("jaw_width_proxy")),
        "ratio_cheekbone_width_to_jaw_width": _safe_divide(raw_measurements.get("cheekbone_width_proxy"), raw_measurements.get("jaw_width_proxy")),
        "ratio_eye_height_to_eye_width": _safe_divide(mean_eye_height, mean_eye_width),
        "ratio_eye_spacing_to_eye_width": _safe_divide(raw_measurements.get("intercanthal_distance"), mean_eye_width),
        "ratio_eye_spacing_to_face_width": _safe_divide(raw_measurements.get("intercanthal_distance"), face_width),
        "ratio_nose_length_to_face_height": _safe_divide(raw_measurements.get("nose_length"), face_height),
        "ratio_philtrum_to_lower_face_height": _safe_divide(raw_measurements.get("philtrum_height"), lower),
        "ratio_upper_lip_to_lower_lip": _safe_divide(raw_measurements.get("upper_lip_height"), raw_measurements.get("lower_lip_height")),
        "ratio_mouth_to_chin": _safe_divide(mouth_width, raw_measurements.get("mouth_center_to_chin_height")),
        "ratio_nose_to_philtrum": _safe_divide(raw_measurements.get("nose_length"), raw_measurements.get("philtrum_height")),
        "ratio_nose_width_to_mouth_width": _safe_divide(raw_measurements.get("nose_width"), mouth_width),
        "lip_thickness_ratio": _safe_divide(
            (raw_measurements.get("upper_lip_height") or 0.0) + (raw_measurements.get("lower_lip_height") or 0.0),
            mouth_width,
        ),
        "chin_taper_ratio": _safe_divide(raw_measurements.get("chin_width_proxy"), raw_measurements.get("jaw_width_proxy")),
    }
    mp_face_width = raw_measurements.get("mediapipe_face_oval_width")
    mp_face_height = raw_measurements.get("mediapipe_face_oval_height")
    mp_face_area_proxy = (
        float(mp_face_width) * float(mp_face_height)
        if mp_face_width is not None and mp_face_height is not None
        else None
    )
    mp_contour_perimeter_proxy = _sum(
        [
            raw_measurements.get("mediapipe_left_face_contour_length_proxy"),
            raw_measurements.get("mediapipe_right_face_contour_length_proxy"),
            raw_measurements.get("mediapipe_lower_face_contour_length_proxy"),
        ]
    )
    mp_eye_width_mean = _mean(
        [raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")]
    )
    mp_eye_height_mean = _mean(
        [raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")]
    )
    mp_brow_to_eye_mean = _mean(
        [raw_measurements.get("left_brow_to_eye_distance"), raw_measurements.get("right_brow_to_eye_distance")]
    )
    mp_cheek_to_jaw_ratio = _safe_divide(
        raw_measurements.get("mediapipe_cheekbone_width_proxy"),
        raw_measurements.get("mediapipe_jaw_width_proxy"),
    )
    mp_eye_to_nose_height = _safe_divide(
        raw_measurements.get("raw_eye_to_nose_px"),
        raw_measurements.get("mediapipe_face_oval_height"),
    )
    mp_nose_base_to_mouth_to_lower_face = _safe_divide(
        raw_measurements.get("nose_base_to_mouth_center_height"),
        raw_measurements.get("lower_third_proxy"),
    )
    result.update(
        {
            "mediapipe_face_oval_aspect_ratio": _safe_divide(raw_measurements.get("mediapipe_face_oval_width"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_left_eye_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_right_eye_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_right_eye_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_mean_eye_height_to_width": _safe_divide(
                _mean([raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")]),
                _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")]),
            ),
            "mediapipe_nose_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_nose_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_nose_length_to_face_height": _safe_divide(raw_measurements.get("mediapipe_nose_length"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_mouth_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_mouth_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_total_lip_height_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_total_lip_height"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_cheekbone_width_to_jaw_ratio": _safe_divide(raw_measurements.get("mediapipe_cheekbone_width_proxy"), raw_measurements.get("mediapipe_jaw_width_proxy")),
            "mediapipe_midface_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_midface_width_proxy"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_midface_height_to_face_height": _safe_divide(raw_measurements.get("mediapipe_midface_height_proxy"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_jaw_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_jaw_width_proxy"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_chin_width_to_jaw_width": _safe_divide(raw_measurements.get("mediapipe_chin_width_proxy"), raw_measurements.get("mediapipe_jaw_width_proxy")),
            "mediapipe_jaw_taper_ratio": _safe_divide(raw_measurements.get("mediapipe_jaw_width_proxy"), raw_measurements.get("mediapipe_cheekbone_width_proxy")),
            "mediapipe_chin_taper_ratio": _safe_divide(raw_measurements.get("mediapipe_chin_width_proxy"), raw_measurements.get("mediapipe_jaw_width_proxy")),
            "mediapipe_lip_area_ratio": _safe_divide(
                raw_measurements.get("mediapipe_upper_lip_area_proxy"),
                raw_measurements.get("mediapipe_lower_lip_area_proxy"),
            ),
            "mediapipe_face_height_to_width": _safe_divide(raw_measurements.get("mediapipe_face_oval_height"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_face_width_to_height": _safe_divide(raw_measurements.get("mediapipe_face_oval_width"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_lower_face_height_to_face_height": _safe_divide(raw_measurements.get("mediapipe_chin_height_proxy"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_midface_to_lower_face_height_ratio": _safe_divide(raw_measurements.get("mediapipe_midface_height_proxy"), raw_measurements.get("mediapipe_chin_height_proxy")),
            "mediapipe_face_contour_left_right_length_ratio": _safe_divide(raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")),
            "mediapipe_face_contour_asymmetry_ratio": _safe_divide(
                _abs_diff(raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")),
                _mean([raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")]),
            ),
            "mediapipe_left_eye_aspect_ratio": _safe_divide(raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_left_eye_width")),
            "mediapipe_right_eye_aspect_ratio": _safe_divide(raw_measurements.get("mediapipe_right_eye_height"), raw_measurements.get("mediapipe_right_eye_width")),
            "mediapipe_mean_eye_aspect_ratio": _safe_divide(
                _mean([raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")]),
                _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")]),
            ),
            "mediapipe_left_eye_height_to_face_height": _safe_divide(raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_right_eye_height_to_face_height": _safe_divide(raw_measurements.get("mediapipe_right_eye_height"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_mean_eye_height_to_face_height": _safe_divide(
                _mean([raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")]),
                raw_measurements.get("mediapipe_face_oval_height"),
            ),
            "mediapipe_intercanthal_to_face_width": _safe_divide(raw_measurements.get("intercanthal_distance"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_outercanthal_to_face_width": _safe_divide(raw_measurements.get("outercanthal_distance"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_intercanthal_to_eye_width": _safe_divide(raw_measurements.get("intercanthal_distance"), _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")])),
            "mediapipe_eye_spacing_balance_index": _safe_divide(
                _abs_diff(raw_measurements.get("intercanthal_distance"), _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")])),
                _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")]),
            ),
            "mediapipe_left_brow_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_right_brow_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_right_brow_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_mean_brow_width_to_face_width": _safe_divide(_mean([raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_right_brow_width")]), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_brow_width_symmetry_ratio": _safe_divide(
                _abs_diff(raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_right_brow_width")),
                _mean([raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_right_brow_width")]),
            ),
            "mediapipe_brow_arch_symmetry_ratio": _safe_divide(
                _abs_diff(raw_measurements.get("mediapipe_left_brow_arch_height_proxy"), raw_measurements.get("mediapipe_right_brow_arch_height_proxy")),
                _mean([raw_measurements.get("mediapipe_left_brow_arch_height_proxy"), raw_measurements.get("mediapipe_right_brow_arch_height_proxy")]),
            ),
            "mediapipe_nose_base_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_nose_base_width"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_nose_bridge_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_nose_bridge_width_proxy"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_nose_width_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_nose_width"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_mouth_width_to_nose_width": _safe_divide(raw_measurements.get("mediapipe_mouth_width"), raw_measurements.get("mediapipe_nose_width")),
            "mediapipe_nose_width_to_intercanthal": _safe_divide(raw_measurements.get("mediapipe_nose_width"), raw_measurements.get("intercanthal_distance")),
            "mediapipe_nose_base_width_to_intercanthal": _safe_divide(raw_measurements.get("mediapipe_nose_base_width"), raw_measurements.get("intercanthal_distance")),
            "mediapipe_nose_midline_offset_to_face_width": _safe_divide(raw_measurements.get("mediapipe_nose_midline_offset"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_mouth_width_to_jaw_width": _safe_divide(raw_measurements.get("mediapipe_mouth_width"), raw_measurements.get("mediapipe_jaw_width_proxy")),
            "mediapipe_mouth_width_to_cheekbone_width": _safe_divide(raw_measurements.get("mediapipe_mouth_width"), raw_measurements.get("mediapipe_cheekbone_width_proxy")),
            "mediapipe_mouth_width_to_intercanthal": _safe_divide(raw_measurements.get("mediapipe_mouth_width"), raw_measurements.get("intercanthal_distance")),
            "mediapipe_mouth_opening_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_mouth_height"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_upper_lip_height_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_upper_lip_height"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_lower_lip_height_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_lower_lip_height"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_upper_lip_to_lower_lip_height_ratio": _safe_divide(raw_measurements.get("mediapipe_upper_lip_height"), raw_measurements.get("mediapipe_lower_lip_height")),
            "mediapipe_lip_fullness_index": _safe_divide(raw_measurements.get("mediapipe_total_lip_height"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_cupid_bow_width_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_cupid_bow_width_proxy"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_cupid_bow_depth_to_upper_lip_height": _safe_divide(raw_measurements.get("mediapipe_cupid_bow_depth_proxy"), raw_measurements.get("mediapipe_upper_lip_height")),
            "mediapipe_cheekbone_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_cheekbone_width_proxy"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_cheekbone_width_to_mouth_width": _safe_divide(raw_measurements.get("mediapipe_cheekbone_width_proxy"), raw_measurements.get("mediapipe_mouth_width")),
            "mediapipe_cheekbone_width_to_nose_width": _safe_divide(raw_measurements.get("mediapipe_cheekbone_width_proxy"), raw_measurements.get("mediapipe_nose_width")),
            "mediapipe_midface_width_to_jaw_width": _safe_divide(raw_measurements.get("mediapipe_midface_width_proxy"), raw_measurements.get("mediapipe_jaw_width_proxy")),
            "mediapipe_jaw_width_to_cheekbone_width": _safe_divide(raw_measurements.get("mediapipe_jaw_width_proxy"), raw_measurements.get("mediapipe_cheekbone_width_proxy")),
            "mediapipe_chin_width_to_face_width": _safe_divide(raw_measurements.get("mediapipe_chin_width_proxy"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_chin_height_to_face_height": _safe_divide(raw_measurements.get("mediapipe_chin_height_proxy"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_chin_height_to_lower_face_height": _safe_divide(raw_measurements.get("mediapipe_chin_height_proxy"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_chin_midline_offset_to_face_width": _safe_divide(raw_measurements.get("mediapipe_chin_midline_offset"), raw_measurements.get("mediapipe_face_oval_width")),
            "mediapipe_lower_face_width_to_midface_width": _safe_divide(raw_measurements.get("mediapipe_jaw_width_proxy"), raw_measurements.get("mediapipe_midface_width_proxy")),
            "mediapipe_upper_to_middle_third_ratio": _safe_divide(raw_measurements.get("upper_third_proxy"), raw_measurements.get("middle_third_proxy")),
            "mediapipe_middle_to_lower_third_ratio": _safe_divide(raw_measurements.get("middle_third_proxy"), raw_measurements.get("lower_third_proxy")),
            "mediapipe_upper_to_lower_third_ratio": _safe_divide(raw_measurements.get("upper_third_proxy"), raw_measurements.get("lower_third_proxy")),
            "mediapipe_thirds_balance_index": thirds_balance_index,
            "mediapipe_nose_to_mouth_to_face_height": _safe_divide(raw_measurements.get("nose_base_to_mouth_center_height"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_eye_to_mouth_to_face_height": _safe_divide(raw_measurements.get("raw_eye_to_mouth_px"), raw_measurements.get("mediapipe_face_oval_height")),
            "mediapipe_mouth_to_chin_to_lower_face": _safe_divide(raw_measurements.get("mouth_center_to_chin_height"), raw_measurements.get("lower_third_proxy")),
            "mediapipe_face_oval_area_proxy": mp_face_area_proxy,
            "mediapipe_face_oval_compactness_proxy": _safe_divide(
                4.0 * math.pi * float(mp_face_area_proxy) if mp_face_area_proxy is not None else None,
                (float(mp_contour_perimeter_proxy) ** 2) if mp_contour_perimeter_proxy is not None else None,
            ),
            "mediapipe_face_taper_index": _safe_divide(
                raw_measurements.get("mediapipe_cheekbone_width_proxy"),
                raw_measurements.get("mediapipe_jaw_width_proxy"),
            ),
            "mediapipe_lower_face_taper_index": _safe_divide(
                raw_measurements.get("mediapipe_jaw_width_proxy"),
                raw_measurements.get("mediapipe_chin_width_proxy"),
            ),
            "mediapipe_upper_face_height_to_face_height": _safe_divide(
                raw_measurements.get("upper_third_proxy"),
                raw_measurements.get("mediapipe_face_oval_height"),
            ),
            "mediapipe_forehead_proxy_to_face_height": _safe_divide(
                raw_measurements.get("raw_forehead_height_proxy_px"),
                raw_measurements.get("mediapipe_face_oval_height"),
            ),
            "mediapipe_mean_eye_width_to_face_width": _safe_divide(
                mp_eye_width_mean,
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_eye_outer_distance_to_face_width": _safe_divide(
                raw_measurements.get("outercanthal_distance"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_left_eye_area_to_face_area_proxy": _safe_divide(
                raw_measurements.get("mediapipe_left_eye_area_proxy"),
                mp_face_area_proxy,
            ),
            "mediapipe_right_eye_area_to_face_area_proxy": _safe_divide(
                raw_measurements.get("mediapipe_right_eye_area_proxy"),
                mp_face_area_proxy,
            ),
            "mediapipe_eye_area_symmetry_ratio": _safe_divide(
                _abs_diff(
                    raw_measurements.get("mediapipe_left_eye_area_proxy"),
                    raw_measurements.get("mediapipe_right_eye_area_proxy"),
                ),
                _mean(
                    [
                        raw_measurements.get("mediapipe_left_eye_area_proxy"),
                        raw_measurements.get("mediapipe_right_eye_area_proxy"),
                    ]
                ),
            ),
            "mediapipe_eye_width_symmetry_ratio": _safe_divide(
                _abs_diff(
                    raw_measurements.get("mediapipe_left_eye_width"),
                    raw_measurements.get("mediapipe_right_eye_width"),
                ),
                mp_eye_width_mean,
            ),
            "mediapipe_eye_height_symmetry_ratio": _safe_divide(
                _abs_diff(
                    raw_measurements.get("mediapipe_left_eye_height"),
                    raw_measurements.get("mediapipe_right_eye_height"),
                ),
                mp_eye_height_mean,
            ),
            "mediapipe_left_brow_to_eye_distance": raw_measurements.get("left_brow_to_eye_distance"),
            "mediapipe_right_brow_to_eye_distance": raw_measurements.get("right_brow_to_eye_distance"),
            "mediapipe_brow_to_eye_distance_symmetry": _safe_divide(
                _abs_diff(
                    raw_measurements.get("left_brow_to_eye_distance"),
                    raw_measurements.get("right_brow_to_eye_distance"),
                ),
                mp_brow_to_eye_mean,
            ),
            "mediapipe_brow_to_eye_distance_to_eye_height": _safe_divide(
                mp_brow_to_eye_mean,
                mp_eye_height_mean,
            ),
            "mediapipe_left_brow_arch_to_eye_height": _safe_divide(
                raw_measurements.get("mediapipe_left_brow_arch_height_proxy"),
                raw_measurements.get("mediapipe_left_eye_height"),
            ),
            "mediapipe_right_brow_arch_to_eye_height": _safe_divide(
                raw_measurements.get("mediapipe_right_brow_arch_height_proxy"),
                raw_measurements.get("mediapipe_right_eye_height"),
            ),
            "mediapipe_brow_inner_distance_to_face_width": _safe_divide(
                raw_measurements.get("intercanthal_distance"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_brow_outer_distance_to_face_width": _safe_divide(
                raw_measurements.get("outercanthal_distance"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_nose_length_to_midface_height": _safe_divide(
                raw_measurements.get("mediapipe_nose_length"),
                raw_measurements.get("mediapipe_midface_height_proxy"),
            ),
            "mediapipe_nose_length_to_lower_face_height": _safe_divide(
                raw_measurements.get("mediapipe_nose_length"),
                raw_measurements.get("lower_third_proxy"),
            ),
            "mediapipe_nose_symmetry_to_face_width": _safe_divide(
                raw_measurements.get("mediapipe_nose_midline_offset"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_nose_tip_to_upper_lip_to_lower_face": _safe_divide(
                raw_measurements.get("mediapipe_nose_length"),
                raw_measurements.get("lower_third_proxy"),
            ),
            "mediapipe_upper_lip_area_to_lower_lip_area_ratio": _safe_divide(
                raw_measurements.get("mediapipe_upper_lip_area_proxy"),
                raw_measurements.get("mediapipe_lower_lip_area_proxy"),
            ),
            "mediapipe_lip_area_to_face_area_proxy": _safe_divide(
                _sum(
                    [
                        raw_measurements.get("mediapipe_upper_lip_area_proxy"),
                        raw_measurements.get("mediapipe_lower_lip_area_proxy"),
                    ]
                ),
                mp_face_area_proxy,
            ),
            "mediapipe_mouth_corner_y_diff_to_mouth_width": _safe_divide(
                raw_measurements.get("left_right_mouth_corner_y_diff"),
                raw_measurements.get("mediapipe_mouth_width"),
            ),
            "mediapipe_cheekbone_width_to_jaw_width": mp_cheek_to_jaw_ratio,
            "mediapipe_cheekbone_height_to_face_height": _safe_divide(
                raw_measurements.get("mediapipe_midface_height_proxy"),
                raw_measurements.get("mediapipe_face_oval_height"),
            ),
            "mediapipe_midface_to_lower_face_height_ratio": _safe_divide(
                raw_measurements.get("mediapipe_midface_height_proxy"),
                raw_measurements.get("lower_third_proxy"),
            ),
            "mediapipe_cheekbone_prominence_proxy": _safe_divide(
                raw_measurements.get("mediapipe_cheekbone_width_proxy"),
                _mean(
                    [
                        raw_measurements.get("mediapipe_jaw_width_proxy"),
                        raw_measurements.get("mediapipe_mouth_width"),
                    ]
                ),
            ),
            "mediapipe_jaw_squareness_proxy": _safe_divide(
                raw_measurements.get("mediapipe_jaw_width_proxy"),
                raw_measurements.get("mediapipe_chin_height_proxy"),
            ),
            "mediapipe_lower_face_width_to_face_width": _safe_divide(
                raw_measurements.get("mediapipe_jaw_width_proxy"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_jaw_left_right_length_ratio": _safe_divide(
                raw_measurements.get("mediapipe_left_face_contour_length_proxy"),
                raw_measurements.get("mediapipe_right_face_contour_length_proxy"),
            ),
            "mediapipe_horizontal_fifths_balance_proxy": _safe_divide(
                raw_measurements.get("outercanthal_distance"),
                raw_measurements.get("mediapipe_face_oval_width"),
            ),
            "mediapipe_eye_to_nose_to_face_height": mp_eye_to_nose_height,
            "mediapipe_nose_base_to_mouth_to_lower_face": mp_nose_base_to_mouth_to_lower_face,
        }
    )
    return result

def extract_symmetry_features(
    raw_measurements: dict[str, float | None],
    normalized_measurements: dict[str, float | None],
) -> dict[str, float | None]:
    eye_distance = raw_measurements.get("raw_eye_distance_px")
    mouth_width = raw_measurements.get("raw_mouth_width_px")
    eye_delta = _safe_divide(raw_measurements.get("left_right_eye_to_nose_diff"), eye_distance)
    mouth_delta = _safe_divide(raw_measurements.get("left_right_mouth_to_nose_diff"), mouth_width)

    offsets = [
        normalized_measurements.get("ratio_nose_center_offset_to_face_width"),
        normalized_measurements.get("ratio_mouth_center_offset_to_face_width"),
        _safe_divide(raw_measurements.get("chin_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("philtrum_midline_offset_proxy"), raw_measurements.get("raw_face_width_px")),
    ]
    present_offsets = [float(value) for value in offsets if value is not None]
    center_offset_ratio = float(sum(present_offsets) / len(present_offsets)) if present_offsets else None

    brow_symmetry_ratio = None
    mean_brow_to_eye = raw_measurements.get("mean_brow_to_eye_distance")
    if mean_brow_to_eye is not None and mean_brow_to_eye > 0:
        brow_symmetry_ratio = max(
            0.0,
            1.0 - float(raw_measurements.get("left_right_brow_height_diff") or 0.0) / float(mean_brow_to_eye),
        )

    midline_alignment_index = None
    if present_offsets:
        clipped = [max(0.0, min(1.0, 1.0 - float(value))) for value in present_offsets]
        midline_alignment_index = float(sum(clipped) / len(clipped))

    upper_components = [
        eye_delta,
        _safe_divide(raw_measurements.get("left_right_eye_y_diff"), raw_measurements.get("raw_face_height_px")),
        _safe_divide(raw_measurements.get("left_right_eye_width_diff"), raw_measurements.get("mean_eye_width")),
        _safe_divide(raw_measurements.get("left_right_eye_height_diff"), raw_measurements.get("mean_eye_height")),
        (1.0 - brow_symmetry_ratio) if brow_symmetry_ratio is not None else None,
        _safe_divide(raw_measurements.get("nose_tip_midline_offset"), raw_measurements.get("raw_face_width_px")),
    ]
    lower_components = [
        mouth_delta,
        _safe_divide(raw_measurements.get("left_right_mouth_corner_y_diff"), raw_measurements.get("raw_face_height_px")),
        _safe_divide(raw_measurements.get("mouth_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("chin_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("philtrum_midline_offset_proxy"), raw_measurements.get("raw_face_width_px")),
    ]

    def _symmetry_index_from_components(values: list[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        if not present:
            return None
        clipped = [max(0.0, min(1.0, 1.0 - value)) for value in present]
        return float(sum(clipped) / len(clipped))

    symmetry_components = [value for value in (eye_delta, mouth_delta, center_offset_ratio) if value is not None]
    symmetry_index = _symmetry_index_from_components(symmetry_components)

    result = {
        "sym_eye_nose_delta_ratio": eye_delta,
        "sym_mouth_nose_delta_ratio": mouth_delta,
        "sym_horizontal_center_offset_ratio": center_offset_ratio,
        "symmetry_index": symmetry_index,
        "brow_symmetry_ratio": brow_symmetry_ratio,
        "midline_alignment_index": midline_alignment_index,
        "upper_face_symmetry_index": _symmetry_index_from_components(upper_components),
        "lower_face_symmetry_index": _symmetry_index_from_components(lower_components),
    }
    mp_eye_width_mean = _mean([raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")])
    mp_eye_height_mean = _mean([raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")])
    mp_eye_area_mean = _mean([raw_measurements.get("mediapipe_left_eye_area_proxy"), raw_measurements.get("mediapipe_right_eye_area_proxy")])
    mp_brow_width_mean = _mean([raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_right_brow_width")])
    mp_brow_arch_mean = _mean([raw_measurements.get("mediapipe_left_brow_arch_height_proxy"), raw_measurements.get("mediapipe_right_brow_arch_height_proxy")])
    mp_contour_mean = _mean([raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")])
    mp_lip_height_mean = _mean([raw_measurements.get("mediapipe_upper_lip_height"), raw_measurements.get("mediapipe_lower_lip_height")])
    mp_face_width = raw_measurements.get("mediapipe_face_oval_width")
    mp_eye_components = [
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_eye_width"), raw_measurements.get("mediapipe_right_eye_width")), mp_eye_width_mean),
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_eye_height"), raw_measurements.get("mediapipe_right_eye_height")), mp_eye_height_mean),
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_eye_area_proxy"), raw_measurements.get("mediapipe_right_eye_area_proxy")), mp_eye_area_mean),
    ]
    mp_brow_components = [
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_brow_width"), raw_measurements.get("mediapipe_right_brow_width")), mp_brow_width_mean),
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_brow_arch_height_proxy"), raw_measurements.get("mediapipe_right_brow_arch_height_proxy")), mp_brow_arch_mean),
        _safe_divide(
            _minimal_axis_angle_difference_deg(raw_measurements.get("mediapipe_left_brow_angle_deg"), raw_measurements.get("mediapipe_right_brow_angle_deg")),
            45.0,
        ),
    ]
    mp_lip_components = [
        _safe_divide(_abs_diff(raw_measurements.get("mediapipe_upper_lip_height"), raw_measurements.get("mediapipe_lower_lip_height")), mp_lip_height_mean),
        _safe_divide(raw_measurements.get("mediapipe_chin_midline_offset"), mp_face_width),
    ]
    result.update(
        {
            "mediapipe_eye_shape_symmetry_index": _symmetry_index_from_components(mp_eye_components),
            "mediapipe_brow_symmetry_index": _symmetry_index_from_components(mp_brow_components),
            "mediapipe_nose_symmetry_index": _symmetry_index_from_components([
                _safe_divide(raw_measurements.get("mediapipe_nose_midline_offset"), mp_face_width),
                _safe_divide(_abs_diff(raw_measurements.get("mediapipe_nose_width"), raw_measurements.get("mediapipe_nose_base_width")), raw_measurements.get("mediapipe_nose_width")),
            ]),
            "mediapipe_lip_symmetry_index": _symmetry_index_from_components(mp_lip_components),
            "mediapipe_face_contour_symmetry_index": _symmetry_index_from_components([
                _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")), mp_contour_mean),
                _safe_divide(raw_measurements.get("mediapipe_chin_midline_offset"), mp_face_width),
            ]),
        }
    )
    result.update(
        {
            "mediapipe_eye_symmetry_composite": result.get("mediapipe_eye_shape_symmetry_index"),
            "mediapipe_brow_symmetry_composite": result.get("mediapipe_brow_symmetry_index"),
            "mediapipe_nose_symmetry_composite": result.get("mediapipe_nose_symmetry_index"),
            "mediapipe_lip_symmetry_composite": result.get("mediapipe_lip_symmetry_index"),
            "mediapipe_jaw_chin_symmetry_composite": _symmetry_index_from_components([
                _safe_divide(raw_measurements.get("mediapipe_chin_midline_offset"), mp_face_width),
                _safe_divide(_abs_diff(raw_measurements.get("mediapipe_left_face_contour_length_proxy"), raw_measurements.get("mediapipe_right_face_contour_length_proxy")), mp_contour_mean),
            ]),
            "mediapipe_face_contour_symmetry_composite": result.get("mediapipe_face_contour_symmetry_index"),
        }
    )
    result["mediapipe_global_symmetry_index"] = _mean(
        [
            result.get("mediapipe_eye_symmetry_composite"),
            result.get("mediapipe_brow_symmetry_composite"),
            result.get("mediapipe_nose_symmetry_composite"),
            result.get("mediapipe_lip_symmetry_composite"),
            result.get("mediapipe_jaw_chin_symmetry_composite"),
            result.get("mediapipe_face_contour_symmetry_composite"),
        ]
    )
    return result

def extract_angle_features(
    landmarks: list[list[float]] | None,
    row: dict[str, Any],
    landmarks_dense: list[list[float]] | None = None,
    bbox: list[float] | None = None,
    mediapipe_landmarks: list[list[float]] | None = None,
) -> dict[str, float | None]:
    result = empty_feature_dict(ANGLE_FEATURE_COLUMNS)
    result["pose_yaw_deg"] = _to_float(row.get("yaw"))
    result["pose_pitch_deg"] = _to_float(row.get("pitch"))
    result["pose_roll_deg"] = _to_float(row.get("roll"))

    if not landmarks or len(landmarks) < 5:
        return result

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    points = np.asarray(landmarks, dtype=np.float64)
    left_eye = context.get("left_eye_center") if isinstance(context.get("left_eye_center"), np.ndarray) else points[0]
    right_eye = context.get("right_eye_center") if isinstance(context.get("right_eye_center"), np.ndarray) else points[1]
    nose = context.get("nose_tip") if isinstance(context.get("nose_tip"), np.ndarray) else points[2]
    left_mouth = context.get("left_mouth") if isinstance(context.get("left_mouth"), np.ndarray) else points[3]
    right_mouth = context.get("right_mouth") if isinstance(context.get("right_mouth"), np.ndarray) else points[4]
    eye_center = context.get("eye_center") if isinstance(context.get("eye_center"), np.ndarray) else (left_eye + right_eye) / 2.0
    mouth_center = context.get("mouth_center") if isinstance(context.get("mouth_center"), np.ndarray) else (left_mouth + right_mouth) / 2.0
    left_brow = context.get("left_brow") if isinstance(context.get("left_brow"), np.ndarray) else None
    right_brow = context.get("right_brow") if isinstance(context.get("right_brow"), np.ndarray) else None
    nose_base_center = context.get("nose_base_center") if isinstance(context.get("nose_base_center"), np.ndarray) else None
    chin_center = context.get("chin_center") if isinstance(context.get("chin_center"), np.ndarray) else None
    jaw_left = context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None
    jaw_right = context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None

    eye_line = _vector_angle_degrees(left_eye, right_eye)
    mouth_line = _vector_angle_degrees(left_mouth, right_mouth)
    nose_axis = _normalize_axis_angle_deg(_vector_angle_degrees(nose_base_center, nose))
    philtrum_axis = _normalize_axis_angle_deg(_vector_angle_degrees(nose_base_center, mouth_center))
    chin_axis = _normalize_axis_angle_deg(_vector_angle_degrees(mouth_center, chin_center))
    brow_line = _vector_angle_degrees(left_brow, right_brow)
    jaw_left_angle = _vector_angle_degrees(jaw_left, chin_center)
    jaw_right_angle = _vector_angle_degrees(chin_center, jaw_right)

    result.update(
        {
            "angle_eye_line_deg": eye_line,
            "angle_mouth_line_deg": mouth_line,
            "angle_eye_center_to_mouth_center_deg": _vector_angle_degrees(eye_center, mouth_center),
            "angle_nose_to_mouth_center_deg": _vector_angle_degrees(nose, mouth_center),
            "canthal_tilt_deg": eye_line,
            "nose_axis_angle_deg": nose_axis,
            "philtrum_axis_angle_deg": philtrum_axis,
            "chin_axis_angle_deg": chin_axis,
            "brow_line_angle_deg": brow_line,
            "jaw_line_left_angle_proxy": jaw_left_angle,
            "jaw_line_right_angle_proxy": jaw_right_angle,
            "jaw_angle_asymmetry": _mirror_angle_asymmetry_deg(jaw_left_angle, jaw_right_angle),
            "face_centerline_vs_nose_angle": _minimal_axis_angle_difference_deg(nose_axis, 90.0),
            "eye_line_vs_mouth_line_diff": _minimal_axis_angle_difference_deg(eye_line, mouth_line),
            "nose_to_chin_axis_diff": _minimal_axis_angle_difference_deg(nose_axis, chin_axis),
            "lower_face_tilt_proxy": chin_axis,
        }
    )
    validation = context.get("semantic_validation")
    if isinstance(validation, dict):
        unreliable_features = {
            str(name)
            for name in validation.get("unreliable_features", [])
            if isinstance(name, str)
        }
        for feature_name in unreliable_features:
            if feature_name in result:
                result[feature_name] = None
    mediapipe_context = _derive_mediapipe_context(mediapipe_landmarks)
    if mediapipe_context:
        left_brow_mp = mediapipe_context.get("left_brow")
        right_brow_mp = mediapipe_context.get("right_brow")
        face_center_mp = mediapipe_context.get("face_center")
        mouth_left_mp = mediapipe_context.get("mouth_left")
        mouth_right_mp = mediapipe_context.get("mouth_right")
        nose_bridge_top_mp = mediapipe_context.get("nose_bridge_top")
        nose_tip_mp = mediapipe_context.get("nose_tip")
        left_eye_outer_mp = mediapipe_context.get("left_eye_outer")
        left_eye_inner_mp = mediapipe_context.get("left_eye_inner")
        right_eye_inner_mp = mediapipe_context.get("right_eye_inner")
        right_eye_outer_mp = mediapipe_context.get("right_eye_outer")
        left_brow_inner_mp, left_brow_outer_mp = _brow_endpoints(left_brow_mp if isinstance(left_brow_mp, np.ndarray) else None, face_center_mp if isinstance(face_center_mp, np.ndarray) else None)
        right_brow_inner_mp, right_brow_outer_mp = _brow_endpoints(right_brow_mp if isinstance(right_brow_mp, np.ndarray) else None, face_center_mp if isinstance(face_center_mp, np.ndarray) else None)
        left_brow_angle = _normalize_axis_angle_deg(_vector_angle_degrees(left_brow_inner_mp, left_brow_outer_mp))
        right_brow_angle = _normalize_axis_angle_deg(_vector_angle_degrees(right_brow_inner_mp, right_brow_outer_mp))
        left_canthal_tilt = _vector_angle_degrees(left_eye_outer_mp, left_eye_inner_mp)
        right_canthal_tilt = _vector_angle_degrees(right_eye_inner_mp, right_eye_outer_mp)
        eye_line_angle_mp = _vector_angle_degrees(left_eye_outer_mp, right_eye_outer_mp)
        nose_axis_angle_mp = _normalize_axis_angle_deg(_vector_angle_degrees(nose_bridge_top_mp, nose_tip_mp))
        mouth_line_angle_mp = _vector_angle_degrees(mouth_left_mp, mouth_right_mp)
        result.update(
            {
                "mediapipe_left_brow_angle_deg": left_brow_angle,
                "mediapipe_right_brow_angle_deg": right_brow_angle,
                "mediapipe_nose_axis_angle_deg": nose_axis_angle_mp,
                "mediapipe_nose_axis_vs_face_midline_deg": _minimal_axis_angle_difference_deg(
                    nose_axis_angle_mp,
                    90.0,
                ),
                "mediapipe_mouth_line_angle_deg": mouth_line_angle_mp,
                "mediapipe_canthal_tilt_left_deg": left_canthal_tilt,
                "mediapipe_canthal_tilt_right_deg": right_canthal_tilt,
                "mediapipe_eye_line_angle_deg": eye_line_angle_mp,
                "mediapipe_brow_angle_diff_abs": _minimal_axis_angle_difference_deg(
                    left_brow_angle,
                    right_brow_angle,
                ),
                "mediapipe_mouth_line_angle_abs": (
                    abs(float(mouth_line_angle_mp))
                    if mouth_line_angle_mp is not None
                    else None
                ),
            }
        )
    return result


def validate_feature_values(
    feature_values: dict[str, float | None],
    *,
    image_id: str,
) -> None:
    issues: list[str] = []

    for key, value in feature_values.items():
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            issues.append(f"{key}=non_finite")
            continue
        if key in NONNEGATIVE_FEATURE_COLUMNS and numeric < 0.0:
            issues.append(f"{key}=negative")
        if key in UNIT_INTERVAL_COLUMNS and not (0.0 <= numeric <= 1.0):
            issues.append(f"{key}=outside_0_1")
        if key.startswith("angle_") or key.startswith("pose_") or key.endswith("_deg"):
            if abs(numeric) > 180.0:
                issues.append(f"{key}=outside_angle_range")

    if issues:
        raise ValueError(f"Invalid measurement values for image_id={image_id}: {', '.join(sorted(issues))}")


def geometry_feature_columns() -> list[str]:
    return RAW_MEASUREMENT_COLUMNS + NORMALIZED_MEASUREMENT_COLUMNS + SYMMETRY_FEATURE_COLUMNS + ANGLE_FEATURE_COLUMNS


def reference_feature_columns() -> list[str]:
    allowed_statuses = {"keep", "new", "corrected"}
    rows = feature_registry_rows()
    return [
        str(row["feature_name"])
        for row in rows
        if row.get("status") in allowed_statuses and row.get("reference_inclusion", "included") != "excluded"
    ]


def feature_registry_rows() -> list[dict[str, Any]]:
    implemented_features = geometry_feature_columns()
    rows: list[dict[str, Any]] = []
    for feature_name in implemented_features:
        if feature_name in RAW_MEASUREMENT_COLUMNS:
            family = "raw_measurements"
        elif feature_name in NORMALIZED_MEASUREMENT_COLUMNS:
            family = "normalized_ratios"
        elif feature_name in SYMMETRY_FEATURE_COLUMNS:
            family = "symmetry_metrics"
        else:
            family = "angle_metrics"

        if feature_name in REDUNDANT_FEATURES:
            status = "redundant"
        elif feature_name in ALIASES:
            status = "deprecated_but_kept_for_compatibility"
        elif feature_name.startswith(("raw_face_", "raw_eye_", "raw_mouth_", "raw_nose_", "ratio_face_", "ratio_eye_", "sym_")):
            status = "keep"
        else:
            status = "new"

        override = FEATURE_METADATA_OVERRIDES.get(feature_name, {})
        component = override.get("component", _feature_component(feature_name, family))
        units = override.get("units", _feature_units(feature_name, family))
        source = override.get("source", _feature_source(feature_name, family))
        reliability = _feature_reliability(feature_name, family, override)
        normalized_by = override.get("normalized_by", _feature_normalized_by(feature_name, family))
        recommended_for_scorer, exclusion_reason = _recommended_for_scorer(
            feature_name,
            family,
            reliability,
            override,
            override.get("status", status),
        )
        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": family,
                "component": component,
                "status": override.get("status", status),
                "alias_of": ALIASES.get(feature_name),
                "depends_on": list(override.get("depends_on", [])),
                "source_points": list(override.get("source_points", [])),
                "source": source,
                "formula": FEATURE_FORMULAS.get(feature_name),
                "units": units,
                "reliability": reliability,
                "required_points": list(override.get("required_points", override.get("source_points", []))),
                "normalized_by": normalized_by,
                "reference_inclusion": override.get("reference_inclusion", "included"),
                "recommended_for_scorer": recommended_for_scorer,
                "exclusion_reason": exclusion_reason,
                "notes": (
                    override.get("notes")
                    or (
                        "Compatibility field retained for downstream consumers."
                        if status == "deprecated_but_kept_for_compatibility"
                        else ("Exact duplicate retained only to avoid breaking old outputs." if status == "redundant" else None)
                    )
                ),
                "reliability_warning": override.get("reliability_warning"),
            }
        )
    rows.extend(OPTIONAL_MULTI_SOURCE_FEATURES)
    return rows


def feature_audit_payload() -> dict[str, Any]:
    rows = feature_registry_rows()
    status_counts: dict[str, int] = {}
    reliability_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        reliability = str(row.get("reliability", "unknown"))
        reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1

    duplicates = [row for row in rows if row["status"] in {"redundant", "deprecated_but_kept_for_compatibility"}]
    unstable = [row for row in rows if row.get("reliability") in {"unstable", "unavailable_without_extra_model"}]
    scorer_excluded = [row for row in rows if not bool(row.get("recommended_for_scorer", False))]

    return {
        "feature_count": len(rows),
        "status_counts": status_counts,
        "reliability_counts": reliability_counts,
        "duplicates_and_aliases": [
            {
                "feature_name": row["feature_name"],
                "status": row["status"],
                "alias_of": row.get("alias_of"),
                "notes": row.get("notes"),
            }
            for row in duplicates
        ],
        "unstable_features": [
            {
                "feature_name": row["feature_name"],
                "notes": row.get("notes"),
                "reference_inclusion": row.get("reference_inclusion"),
            }
            for row in unstable
        ],
        "scorer_excluded_features": [
            {
                "feature_name": row["feature_name"],
                "reliability": row.get("reliability"),
                "exclusion_reason": row.get("exclusion_reason"),
            }
            for row in scorer_excluded
        ],
        "registry_version": "morphometric_feature_registry_v3_multi_source",
    }
