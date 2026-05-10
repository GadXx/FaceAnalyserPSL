from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_path: str
    output_dir: str | None = None
    group: str | None = None
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
    description: str | None = None


class SemanticValidationResponse(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
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
    raw_landmarks: list[LandmarkPointResponse] = Field(default_factory=list)
    raw_landmarks_debug: list[LandmarkPointResponse] = Field(default_factory=list)
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
    z_scores: dict[str, float | None]
    mahalanobis_distance: float
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


class AnalyzeResponse(BaseModel):
    input: dict[str, Any]
    input_quality: InputQualityResponse
    visualization: VisualizationResponse | None = None
    raw_measurements: dict[str, float | None]
    normalized_ratios: dict[str, float | None]
    symmetry_metrics: dict[str, float | None]
    angle_metrics: dict[str, float | None]
    reference_comparison: ReferenceComparisonResponse
    feature_level_deviations: FeatureLevelDeviationsResponse
    optional_composite_indices: dict[str, float | None]
    transparency: dict[str, Any]
    limitations: list[str]
    artifacts: dict[str, Any]
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
