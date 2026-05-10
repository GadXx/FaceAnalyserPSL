export function getRawLandmarks(visualization) {
  return Array.isArray(visualization?.raw_landmarks) ? visualization.raw_landmarks : [];
}

export function getSemanticPoints(visualization) {
  if (!visualization || typeof visualization !== "object") {
    return {};
  }
  return visualization.semantic_points && typeof visualization.semantic_points === "object"
    ? visualization.semantic_points
    : {};
}

export function getMeasurementOverlaySegments(visualization, key) {
  const overlays = visualization?.measurement_overlays;
  if (!overlays || typeof overlays !== "object") {
    return [];
  }
  const candidate = overlays[key];
  return Array.isArray(candidate) ? candidate : [];
}
