export interface AnalyzeRequestOptions {
  group?: string;
  outputDir?: string;
  preferredExecutionProvider?: string;
  deviceId?: number;
  strictGpu?: boolean;
}

export interface ApiErrorPayload {
  detail?: string | { detail?: string; error_type?: string };
  error_type?: string;
}

export interface DependencyStatus {
  name: string;
  path: string;
  required: boolean;
  available: boolean;
  message?: string | null;
}

export interface HealthResponse {
  status: string;
  can_analyze: boolean;
  project_root: string;
  dependencies: DependencyStatus[];
}

export interface ReferenceGroupEntry {
  combination_name: string;
  group_key: string;
  group_values: Record<string, string>;
  sample_count: number;
  complete_case_count: number;
  feature_count: number;
  statistics_feature_rows: number;
}

export interface ReferenceGroupsResponse {
  status: string;
  reference_model_path: string;
  reference_statistics_path: string;
  groupings: string[][];
  reference_group_count: number;
  groups: ReferenceGroupEntry[];
}

export interface DetectionPoseResponse {
  yaw: number | null;
  pitch: number | null;
  roll: number | null;
}

export interface Point2DResponse {
  x: number;
  y: number;
}

export interface ImageSizeResponse {
  width: number;
  height: number;
}

export interface LandmarkPointResponse {
  index: number;
  x: number;
  y: number;
  group?: string | null;
  name?: string | null;
  source_type?: string | null;
  source_indices?: number[];
  description?: string | null;
}

export interface SemanticPointResponse {
  x: number;
  y: number;
  source_index_106?: number | null;
  source_indices?: number[];
  source_region?: string | null;
  source_type?: string | null;
  validation_status?: string | null;
  point_name?: string | null;
}

export interface SemanticPointDebugResponse {
  point_name: string;
  group?: string | null;
  x: number;
  y: number;
  source_type: string;
  source_indices?: number[];
  source_index_106?: number | null;
  source_region?: string | null;
  validation_status?: string | null;
  description?: string | null;
}

export interface SemanticValidationResponse {
  passed: boolean;
  warnings?: string[];
  failed_checks?: string[];
  unreliable_features?: string[];
}

export interface VisualizationLineResponse {
  key: string;
  start: string;
  end: string;
  group: string;
  kind?: string | null;
  layers?: string[];
  metric_key?: string | null;
  supported_features?: string[];
  label?: string | null;
}

export interface DetectionInfoResponse {
  bbox: number[] | null;
  det_score: number | null;
  pose: DetectionPoseResponse;
  num_faces: number | null;
  detection_backend: string | null;
  blur_variance?: number | null;
  occlusion_score?: number | null;
}

export interface InputQualityResponse {
  processing_status: string | null;
  quality_pass: boolean;
  quality_fail_reason: string | null;
  processing_error_message?: string | null;
  detection: DetectionInfoResponse;
}

export interface VisualizationResponse {
  status: string;
  coordinate_space: string;
  image_role?: string | null;
  image_path?: string | null;
  image_width?: number | null;
  image_height?: number | null;
  image_size?: ImageSizeResponse | null;
  bbox?: number[] | null;
  raw_landmarks?: LandmarkPointResponse[];
  raw_landmarks_debug?: LandmarkPointResponse[];
  semantic_points?: Record<string, SemanticPointResponse> | null;
  semantic_points_debug?: SemanticPointDebugResponse[];
  semantic_validation?: SemanticValidationResponse | null;
  measurement_overlays?: {
    lines?: VisualizationLineResponse[];
    axes?: VisualizationLineResponse[];
    guides?: VisualizationLineResponse[];
  } | null;
  measurement_overlays_debug?: {
    lines?: VisualizationLineResponse[];
    axes?: VisualizationLineResponse[];
    guides?: VisualizationLineResponse[];
  } | null;
  landmarks?: LandmarkPointResponse[];
  anchors?: Record<string, Point2DResponse> | null;
  named_points?: Record<string, Point2DResponse> | null;
  derived_points?: Record<string, Point2DResponse> | null;
  lines?: VisualizationLineResponse[];
  axes?: VisualizationLineResponse[];
  guides?: VisualizationLineResponse[];
  groups?: Record<string, string[]>;
  notes?: string[];
}

export interface FeatureDeviation {
  feature_name: string;
  z_score: number;
  abs_z_score: number;
  direction: string;
}

export interface AnalyzeResponse {
  input: Record<string, unknown>;
  input_quality: InputQualityResponse;
  visualization?: VisualizationResponse | null;
  raw_measurements: Record<string, number | null>;
  normalized_ratios: Record<string, number | null>;
  symmetry_metrics: Record<string, number | null>;
  angle_metrics: Record<string, number | null>;
  reference_comparison: {
    selected_group: {
      group_key: string;
      group_values?: Record<string, string> | null;
      sample_count?: number | null;
      complete_case_count?: number | null;
    };
    z_scores: Record<string, number | null>;
    mahalanobis_distance: number;
    percentile_positions: Record<string, number | null>;
  };
  feature_level_deviations: {
    top_contributing_features: FeatureDeviation[];
    all_feature_deviations: FeatureDeviation[];
  };
  optional_composite_indices: Record<string, number | null>;
  transparency: Record<string, unknown>;
  limitations: string[];
  artifacts: Record<string, unknown>;
  generated_artifacts: {
    report_json_path: string;
    report_markdown_path: string;
    reporting_summary_path: string;
    analysis_image_path?: string | null;
    normalized_frame_image_path?: string | null;
    aligned_image_path?: string | null;
    measurement_image_path?: string | null;
    overlay_image_path?: string | null;
    annotated_image_path?: string | null;
  };
}
