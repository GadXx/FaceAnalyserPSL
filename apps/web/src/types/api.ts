export interface AnalyzeRequestOptions {
  group?: string;
  gender?: string;
  ageGroup?: string;
  race?: string;
  outputDir?: string;
  preferredExecutionProvider?: string;
  deviceId?: number;
  strictGpu?: boolean;
}

export interface PrimaryScore {
  predicted_score: number;
  score_source?: "raw_continuous" | "raw" | "calibrated" | string | null;
  uses_calibration?: boolean | null;
  calibration_used_as_primary?: boolean | null;
  calibration_used_for_diagnostics?: boolean | null;
  display_score?: number | null;
  predicted_score_raw?: number | null;
  raw_predicted_score?: number | null;
  calibrated_predicted_score?: number | null;
  calibration_method?: string | null;
  model?: string | null;
  model_family?: string | null;
  nearest_score_group?: number | null;
  distance_to_nearest_group?: number | null;
  distances_to_groups?: Record<string, number>;
  soft_group_membership?: Record<string, number>;
  score_range?: Array<number | null>;
  observed_training_label_range?: Array<number | null>;
  feature_set_name?: string | null;
  primary_feature_set?: string | null;
  selected_feature_source_counts?: Record<string, number>;
  mediapipe_features_used?: number | null;
  direct_mediapipe_features_used?: number | null;
  mediapipe_derived_reference_features_used?: number | null;
  total_mediapipe_influence_count?: number | null;
  reference_relative_features_used?: number | null;
  expanded_geometry_available?: boolean | null;
  expanded_geometry_used_in_primary?: boolean | null;
  top_mediapipe_contributors?: FeatureContribution[];
  active_feature_source_policy?: Record<string, unknown> | null;
}

export interface PairwiseScoreResult {
  ranking_score?: number | null;
  approx_score?: number | null;
  nearest_score_group?: number | null;
  comparison_count?: number | null;
  confidence?: string | null;
}

export interface OrdinalScoreResult {
  predicted_score?: number | null;
  nearest_score_group?: number | null;
  threshold_probabilities?: Record<string, number>;
}

export interface ScoreClassificationTopGroup {
  group: string;
  probability: number;
}

export interface ClassifierDiagnosticResult {
  predicted_group?: string | null;
  group_probabilities?: Record<string, number>;
  top_k_groups?: ScoreClassificationTopGroup[];
}

export interface ScoreConfidence {
  level: string;
  reasons?: string[];
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
  value?: number | null;
  local_contribution?: number | null;
  global_weight?: number | null;
  component?: string | null;
  anatomical_component?: string | null;
  transform_family?: string | null;
  reference_transform?: string | null;
  base_feature_name?: string | null;
  source?: string | null;
  original_source?: string | null;
  readable_label?: string | null;
}

export interface ComponentContribution {
  contribution: number;
  direction: "positive" | "negative" | "neutral" | string;
  feature_count?: number | null;
  top_features?: FeatureContribution[];
}

export interface LocalExplanation {
  baseline_score?: number | null;
  predicted_score?: number | null;
  explanation_method?: string | null;
  explained_score_kind?: string | null;
  feature_contributions?: FeatureContribution[];
  top_positive_features?: FeatureContribution[];
  top_negative_features?: FeatureContribution[];
  feature_global_weights?: Array<Record<string, unknown>>;
  component_contributions?: Record<string, ComponentContribution>;
  additive_check?: Record<string, unknown>;
}

export interface ScoreReferenceContext {
  source?: string | null;
  stats_path?: string | null;
  filters?: string | Record<string, string> | null;
  requested_group?: string | Record<string, string> | null;
  resolved_group?: string | Record<string, string> | null;
  fallback_used?: boolean | null;
  reference_sample_count?: number | null;
}

export interface ScoringModelInfo {
  primary_model?: string | null;
  primary_feature_set?: string | null;
  uses_mediapipe?: boolean | null;
  uses_reference_relative?: boolean | null;
  weight_method?: string | null;
  selected_feature_count?: number | null;
  selected_component_counts?: Record<string, number>;
  selected_transform_family_counts?: Record<string, number>;
  selected_source_counts?: Record<string, number>;
}

export interface LearnedWeightsPayload {
  component_weights?: Array<Record<string, unknown>>;
  anatomical_component_weights?: Array<Record<string, unknown>>;
  transform_family_weights?: Array<Record<string, unknown>>;
  source_weights?: Array<Record<string, unknown>>;
  top_feature_weights?: Array<Record<string, unknown>>;
}

export interface AuxiliaryScores {
  regression?: Partial<PrimaryScore> | null;
  pairwise?: PairwiseScoreResult | null;
  ordinal?: OrdinalScoreResult | null;
  classifier?: ClassifierDiagnosticResult | null;
  ensemble?: {
    predicted_score?: number | null;
    nearest_score_group?: number | null;
    std?: number | null;
  } | null;
}

export interface ScoreResponse {
  primary_score?: PrimaryScore | null;
  auxiliary_scores?: AuxiliaryScores | null;
  regression: PrimaryScore;
  classification: ClassifierDiagnosticResult;
  pairwise?: PairwiseScoreResult | null;
  ordinal?: OrdinalScoreResult | null;
  final_score?: {
    predicted_score?: number | null;
    nearest_score_group?: number | null;
    confidence?: string | null;
    model_agreement?: Record<string, number | null>;
  } | null;
  local_explanation?: LocalExplanation | null;
  learned_weighting?: Record<string, unknown>;
  learned_weights?: LearnedWeightsPayload | null;
  scoring_model_info?: ScoringModelInfo | null;
  component_contributions?: Record<string, ComponentContribution>;
  explanation_method?: string | null;
  feature_set_name?: string | null;
  selected_feature_source_counts?: Record<string, number>;
  mediapipe_features_used?: number | null;
  direct_mediapipe_features_used?: number | null;
  mediapipe_derived_reference_features_used?: number | null;
  total_mediapipe_influence_count?: number | null;
  reference_relative_features_used?: number | null;
  reference_context?: ScoreReferenceContext | null;
  confidence: ScoreConfidence;
  quality: Record<string, unknown>;
  evaluation_context?: Record<string, unknown>;
  top_contributing_features?: Array<Record<string, unknown>>;
  diagnostic_disagreement?: Record<string, unknown>;
  warnings?: string[];
  generated_artifacts: {
    report_json_path: string;
    report_markdown_path: string;
  };
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
  source_rule?: string | null;
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
  source_rule?: string | null;
  description?: string | null;
}

export interface SemanticFaceExtentsResponse {
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
  center_x: number;
  center_y: number;
  source_type?: string | null;
  source_rule?: string | null;
  source_indices?: number[];
}

export interface SemanticValidationResponse {
  passed: boolean;
  warnings?: string[];
  failed_checks?: string[];
  warning_checks?: string[];
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
  detector_bbox?: number[] | null;
  display_face_box?: number[] | null;
  semantic_face_extents?: SemanticFaceExtentsResponse | null;
  raw_landmarks?: LandmarkPointResponse[];
  raw_landmarks_debug?: LandmarkPointResponse[];
  mediapipe_landmarks?: LandmarkPointResponse[];
  mediapipe_regions?: Record<string, number[]>;
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

export interface ExpandedGeometryFeatureRow {
  feature_name: string;
  value: number | null;
  source?: string | null;
  reliability?: string | null;
  units?: string | null;
  component?: string | null;
  recommended_for_scorer?: boolean | null;
  exclusion_reason?: string | null;
}

export interface ExpandedGeometryResponse {
  available: boolean;
  mediapipe_feature_count: number;
  mediapipe_non_null_feature_count: number;
  reliability_counts?: Record<string, number>;
  groups?: Record<string, ExpandedGeometryFeatureRow[]>;
  unavailable_without_face_parsing?: Array<Record<string, unknown>>;
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
  expanded_geometry?: ExpandedGeometryResponse;
  reference_comparison: {
    selected_group: {
      group_key: string;
      group_values?: Record<string, string> | null;
      sample_count?: number | null;
      complete_case_count?: number | null;
    };
    status?: string;
    warning?: string | null;
    used_feature_names?: string[];
    missing_feature_names?: string[];
    unstable_reference_features?: string[];
    excluded_reference_features?: string[];
    available_feature_count?: number;
    required_feature_count?: number;
    z_scores: Record<string, number | null>;
    mahalanobis_distance: number | null;
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
  preview_image?: {
    clean_image_url?: string | null;
    overlay_layers?: string[];
  };
  overlay?: {
    active_source?: string;
    layers?: string[];
    legacy_layers?: string[];
  };
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
