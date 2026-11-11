from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    group: str | None = None
    gender: str | None = None
    age_group: str | None = None
    race: str | None = None
    preferred_execution_provider: str | None = None
    device_id: int | None = None
    strict_gpu: bool | None = None


class DetectionPoseResponse(BaseModel):
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


class AlignmentInfoResponse(BaseModel):
    attempted: bool = False
    applied: bool = False
    improved_quality: bool = False
    source: str | None = None
    status: str | None = None
    rotation_deg: float | None = None
    transform: dict[str, Any] | None = None
    original_pose: DetectionPoseResponse
    aligned_pose: DetectionPoseResponse
    original_quality_pass: bool | None = None
    original_quality_fail_reason: str | None = None
    aligned_quality_pass: bool | None = None
    aligned_quality_fail_reason: str | None = None


class DetectionInfoResponse(BaseModel):
    bbox: list[float] | None = None
    det_score: float | None = None
    pose: DetectionPoseResponse
    num_faces: int | None = None
    detection_backend: str | None = None
    blur_variance: float | None = None
    occlusion_score: float | None = None


class InputQualityResponse(BaseModel):
    processing_status: str | None = None
    quality_pass: bool
    quality_fail_reason: str | None = None
    processing_error_message: str | None = None
    detection: DetectionInfoResponse
    alignment: AlignmentInfoResponse | None = None


class Point2DResponse(BaseModel):
    x: float
    y: float


class ImageSizeResponse(BaseModel):
    width: int
    height: int


class LandmarkPointResponse(BaseModel):
    index: int
    x: float
    y: float
    group: str | None = None
    name: str | None = None
    source_type: str | None = None
    source_indices: list[int] = Field(default_factory=list)
    description: str | None = None


class SemanticPointResponse(BaseModel):
    x: float
    y: float
    source_index_106: int | None = None
    source_indices: list[int] = Field(default_factory=list)
    source_region: str | None = None
    source_type: str | None = None
    validation_status: str | None = None
    point_name: str | None = None
    source_rule: str | None = None


class SemanticPointDebugResponse(BaseModel):
    point_name: str
    group: str | None = None
    x: float
    y: float
    source_type: str
    source_indices: list[int] = Field(default_factory=list)
    source_index_106: int | None = None
    source_region: str | None = None
    validation_status: str | None = None
    source_rule: str | None = None
    description: str | None = None


class SemanticFaceExtentsResponse(BaseModel):
    left: float
    right: float
    top: float
    bottom: float
    width: float
    height: float
    center_x: float
    center_y: float
    source_type: str | None = None
    source_rule: str | None = None
    source_indices: list[int] = Field(default_factory=list)


class SemanticValidationResponse(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    warning_checks: list[str] = Field(default_factory=list)
    unreliable_features: list[str] = Field(default_factory=list)


class VisualizationLineResponse(BaseModel):
    key: str
    start: str
    end: str
    group: str
    kind: str | None = None
    layers: list[str] = Field(default_factory=list)
    metric_key: str | None = None
    supported_features: list[str] = Field(default_factory=list)
    label: str | None = None


class VisualizationResponse(BaseModel):
    status: str
    coordinate_space: str
    image_role: str | None = None
    image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_size: ImageSizeResponse | None = None
    bbox: list[float] | None = None
    detector_bbox: list[float] | None = None
    display_face_box: list[float] | None = None
    semantic_face_extents: SemanticFaceExtentsResponse | None = None
    raw_landmarks: list[LandmarkPointResponse] = Field(default_factory=list)
    raw_landmarks_debug: list[LandmarkPointResponse] = Field(default_factory=list)
    mediapipe_landmarks: list[LandmarkPointResponse] = Field(default_factory=list)
    mediapipe_regions: dict[str, list[int]] = Field(default_factory=dict)
    semantic_points: dict[str, SemanticPointResponse] | None = None
    semantic_points_debug: list[SemanticPointDebugResponse] = Field(default_factory=list)
    semantic_validation: SemanticValidationResponse | None = None
    measurement_overlays: dict[str, list[VisualizationLineResponse]] = Field(default_factory=dict)
    measurement_overlays_debug: dict[str, list[VisualizationLineResponse]] = Field(default_factory=dict)
    landmarks: list[LandmarkPointResponse] = Field(default_factory=list)
    anchors: dict[str, Point2DResponse] | None = None
    named_points: dict[str, Point2DResponse] | None = None
    derived_points: dict[str, Point2DResponse] | None = None
    lines: list[VisualizationLineResponse] = Field(default_factory=list)
    axes: list[VisualizationLineResponse] = Field(default_factory=list)
    guides: list[VisualizationLineResponse] = Field(default_factory=list)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SelectedGroupResponse(BaseModel):
    group_key: str
    group_values: dict[str, str] | None = None
    sample_count: int | None = None
    complete_case_count: int | None = None


class FeatureDeviationResponse(BaseModel):
    feature_name: str
    z_score: float
    abs_z_score: float
    direction: str


class FeatureLevelDeviationsResponse(BaseModel):
    top_contributing_features: list[FeatureDeviationResponse]
    all_feature_deviations: list[FeatureDeviationResponse]


class ReferenceComparisonResponse(BaseModel):
    selected_group: SelectedGroupResponse
    status: str = "complete"
    warning: str | None = None
    used_feature_names: list[str] = Field(default_factory=list)
    missing_feature_names: list[str] = Field(default_factory=list)
    unstable_reference_features: list[str] = Field(default_factory=list)
    excluded_reference_features: list[str] = Field(default_factory=list)
    available_feature_count: int = 0
    required_feature_count: int = 0
    z_scores: dict[str, float | None]
    mahalanobis_distance: float | None = None
    percentile_positions: dict[str, float | None]


class GeneratedArtifactsResponse(BaseModel):
    report_json_path: str
    report_markdown_path: str
    reporting_summary_path: str
    analysis_image_path: str | None = None
    normalized_frame_image_path: str | None = None
    aligned_image_path: str | None = None
    measurement_image_path: str | None = None
    overlay_image_path: str | None = None
    annotated_image_path: str | None = None


class PreviewImageResponse(BaseModel):
    clean_image_url: str | None = None
    overlay_layers: list[str] = Field(default_factory=list)


class OverlayResponse(BaseModel):
    active_source: str = "mediapipe_only"
    layers: list[str] = Field(default_factory=list)
    legacy_layers: list[str] = Field(default_factory=list)


class ScoreRegressionResult(BaseModel):
    predicted_score: float
    score_source: str | None = None
    uses_calibration: bool | None = None
    calibration_used_as_primary: bool | None = None
    calibration_used_for_diagnostics: bool | None = None
    display_score: float | None = None
    predicted_score_raw: float | None = None
    raw_predicted_score: float | None = None
    calibrated_predicted_score: float | None = None
    calibration_method: str | None = None
    model: str | None = None
    model_family: str | None = None
    nearest_score_group: float | None = None
    distance_to_nearest_group: float | None = None
    distances_to_groups: dict[str, float] = Field(default_factory=dict)
    soft_group_membership: dict[str, float] = Field(default_factory=dict)
    score_range: list[float | None] = Field(default_factory=list)
    observed_training_label_range: list[float | None] = Field(default_factory=list)
    feature_set_name: str | None = None
    selected_feature_source_counts: dict[str, int] = Field(default_factory=dict)
    mediapipe_features_used: int | None = None
    reference_relative_features_used: int | None = None
    primary_feature_set: str | None = None
    expanded_geometry_available: bool | None = None
    expanded_geometry_used_in_primary: bool | None = None
    top_mediapipe_contributors: list[dict[str, Any]] = Field(default_factory=list)
    direct_mediapipe_features_used: int | None = None
    mediapipe_derived_reference_features_used: int | None = None
    total_mediapipe_influence_count: int | None = None
    active_feature_source_policy: dict[str, Any] = Field(default_factory=dict)


class ScoreClassificationTopGroup(BaseModel):
    group: str
    probability: float


class ScoreClassificationResult(BaseModel):
    predicted_group: str | None = None
    group_probabilities: dict[str, float] = Field(default_factory=dict)
    top_k_groups: list[ScoreClassificationTopGroup] = Field(default_factory=list)


class ScoreConfidence(BaseModel):
    level: str
    reasons: list[str] = Field(default_factory=list)


class ScoreFeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float | None = None
    local_contribution: float | None = None
    global_weight: float | None = None
    component: str | None = None
    anatomical_component: str | None = None
    transform_family: str | None = None
    reference_transform: str | None = None
    base_feature_name: str | None = None
    source: str | None = None
    original_source: str | None = None
    readable_label: str | None = None


class ScoreComponentContribution(BaseModel):
    contribution: float
    direction: str
    feature_count: int | None = None
    top_features: list[ScoreFeatureContribution] = Field(default_factory=list)


class ScoreLocalExplanation(BaseModel):
    baseline_score: float | None = None
    predicted_score: float | None = None
    explanation_method: str | None = None
    explained_score_kind: str | None = None
    feature_contributions: list[ScoreFeatureContribution] = Field(default_factory=list)
    top_positive_features: list[ScoreFeatureContribution] = Field(default_factory=list)
    top_negative_features: list[ScoreFeatureContribution] = Field(default_factory=list)
    feature_global_weights: list[dict[str, Any]] = Field(default_factory=list)
    component_contributions: dict[str, ScoreComponentContribution] = Field(default_factory=dict)
    additive_check: dict[str, Any] = Field(default_factory=dict)


class ScoreModelInfo(BaseModel):
    primary_model: str | None = None
    primary_feature_set: str | None = None
    uses_mediapipe: bool | None = None
    uses_reference_relative: bool | None = None
    weight_method: str | None = None
    selected_feature_count: int | None = None
    selected_component_counts: dict[str, int] = Field(default_factory=dict)
    selected_transform_family_counts: dict[str, int] = Field(default_factory=dict)
    selected_source_counts: dict[str, int] = Field(default_factory=dict)


class ScoreLearnedWeights(BaseModel):
    component_weights: list[dict[str, Any]] = Field(default_factory=list)
    anatomical_component_weights: list[dict[str, Any]] = Field(default_factory=list)
    transform_family_weights: list[dict[str, Any]] = Field(default_factory=list)
    source_weights: list[dict[str, Any]] = Field(default_factory=list)
    top_feature_weights: list[dict[str, Any]] = Field(default_factory=list)


class ScoreReferenceContext(BaseModel):
    source: str | None = None
    stats_path: str | None = None
    filters: str | dict[str, Any] | None = None
    requested_group: str | dict[str, Any] | None = None
    resolved_group: str | dict[str, Any] | None = None
    fallback_used: bool | None = None
    reference_sample_count: int | None = None


class ScoreGeneratedArtifactsResponse(BaseModel):
    report_json_path: str
    report_markdown_path: str


class ScoreResponse(BaseModel):
    primary_score: ScoreRegressionResult | None = None
    auxiliary_scores: dict[str, Any] = Field(default_factory=dict)
    regression: ScoreRegressionResult
    classification: ScoreClassificationResult
    pairwise: dict[str, Any] = Field(default_factory=dict)
    ordinal: dict[str, Any] = Field(default_factory=dict)
    final_score: dict[str, Any] = Field(default_factory=dict)
    local_explanation: ScoreLocalExplanation | None = None
    learned_weighting: dict[str, Any] = Field(default_factory=dict)
    learned_weights: ScoreLearnedWeights | None = None
    scoring_model_info: ScoreModelInfo | None = None
    component_contributions: dict[str, ScoreComponentContribution] = Field(default_factory=dict)
    explanation_method: str | None = None
    feature_set_name: str | None = None
    selected_feature_source_counts: dict[str, int] = Field(default_factory=dict)
    mediapipe_features_used: int | None = None
    direct_mediapipe_features_used: int | None = None
    mediapipe_derived_reference_features_used: int | None = None
    total_mediapipe_influence_count: int | None = None
    reference_relative_features_used: int | None = None
    reference_context: ScoreReferenceContext | None = None
    confidence: ScoreConfidence
    quality: dict[str, Any]
    evaluation_context: dict[str, Any] = Field(default_factory=dict)
    top_contributing_features: list[dict[str, Any]] = Field(default_factory=list)
    diagnostic_disagreement: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    generated_artifacts: ScoreGeneratedArtifactsResponse


class AnalyzeResponse(BaseModel):
    input: dict[str, Any]
    input_quality: InputQualityResponse
    visualization: VisualizationResponse | None = None
    raw_measurements: dict[str, float | None]
    normalized_ratios: dict[str, float | None]
    symmetry_metrics: dict[str, float | None]
    angle_metrics: dict[str, float | None]
    expanded_geometry: dict[str, Any] = Field(default_factory=dict)
    reference_comparison: ReferenceComparisonResponse
    feature_level_deviations: FeatureLevelDeviationsResponse
    optional_composite_indices: dict[str, float | None]
    transparency: dict[str, Any]
    limitations: list[str]
    artifacts: dict[str, Any]
    preview_image: PreviewImageResponse | None = None
    overlay: OverlayResponse | None = None
    generated_artifacts: GeneratedArtifactsResponse


class DependencyStatusResponse(BaseModel):
    name: str
    path: str
    required: bool
    available: bool
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    can_analyze: bool
    project_root: str
    dependencies: list[DependencyStatusResponse]


class ReferenceGroupEntryResponse(BaseModel):
    combination_name: str
    group_key: str
    group_values: dict[str, str]
    sample_count: int
    complete_case_count: int
    feature_count: int
    statistics_feature_rows: int


class ReferenceGroupsResponse(BaseModel):
    status: str
    reference_model_path: str
    reference_statistics_path: str
    groupings: list[list[str]]
    reference_group_count: int
    groups: list[ReferenceGroupEntryResponse]


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error description.")
    error_type: str = Field(..., description="Stable error category for client handling.")
