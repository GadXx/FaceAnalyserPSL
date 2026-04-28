import type { LandmarkPointResponse, Point2DResponse, VisualizationLineResponse, VisualizationResponse } from "../types/api";

export function getRawLandmarks(visualization: VisualizationResponse | null | undefined): LandmarkPointResponse[];
export function getSemanticPoints(
  visualization: VisualizationResponse | null | undefined,
): Record<string, Point2DResponse>;
export function getMeasurementOverlaySegments(
  visualization: VisualizationResponse | null | undefined,
  key: "lines" | "axes" | "guides",
): VisualizationLineResponse[];
