export function getRawLandmarks(visualization) {
  return Array.isArray(visualization?.raw_landmarks) ? visualization.raw_landmarks : [];
}

export function getMediapipeLandmarks(visualization) {
  return Array.isArray(visualization?.mediapipe_landmarks) ? visualization.mediapipe_landmarks : [];
}

export function getMediapipeRegions(visualization) {
  if (!visualization || typeof visualization !== "object") {
    return {};
  }
  return visualization.mediapipe_regions && typeof visualization.mediapipe_regions === "object"
    ? visualization.mediapipe_regions
    : {};
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

export function getDisplayFaceBox(visualization, fallbackBBox = null) {
  if (Array.isArray(visualization?.display_face_box) && visualization.display_face_box.length === 4) {
    return visualization.display_face_box;
  }
  if (Array.isArray(visualization?.bbox) && visualization.bbox.length === 4) {
    return visualization.bbox;
  }
  return Array.isArray(fallbackBBox) && fallbackBBox.length === 4 ? fallbackBBox : null;
}
