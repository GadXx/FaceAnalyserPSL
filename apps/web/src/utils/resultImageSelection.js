/**
 * @typedef {{
 *   source: string;
 *   path: string | null;
 *   url: string | null;
 * }} SelectedResultImage
 */

/**
 * @param {string | null | undefined} path
 * @param {number | null | undefined} cacheKey
 * @param {(path: string | null | undefined) => string | null} buildArtifactUrl
 * @returns {string | null}
 */
function artifactUrl(path, cacheKey, buildArtifactUrl) {
  const base = buildArtifactUrl(path ?? null);
  if (!base) {
    return null;
  }
  if (!cacheKey) {
    return base;
  }
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}v=${encodeURIComponent(String(cacheKey))}`;
}

/**
 * @param {any} result
 * @param {number | null | undefined} cacheKey
 * @param {(path: string | null | undefined) => string | null} buildArtifactUrl
 * @returns {SelectedResultImage}
 */
export function selectResultImage(result, cacheKey, buildArtifactUrl) {
  if (!result) {
    return {
      source: "none",
      path: null,
      url: null,
    };
  }

  const generatedArtifacts = result.generated_artifacts ?? {};
  const artifacts = result.artifacts ?? {};
  const visualization = result.visualization ?? {};
  const previewImage = result.preview_image ?? {};

  /** @type {Array<{source: string, path: unknown}>} */
  const candidates = [
    { source: "preview_image.clean_image_url", path: previewImage.clean_image_url },
    { source: "generated_artifacts.analysis_image_path", path: generatedArtifacts.analysis_image_path },
    { source: "visualization.image_path", path: visualization.image_path },
    { source: "generated_artifacts.measurement_image_path", path: generatedArtifacts.measurement_image_path },
    { source: "artifacts.analysis_image_path", path: artifacts.analysis_image_path },
    { source: "artifacts.measurement_image_path", path: artifacts.measurement_image_path },
    { source: "generated_artifacts.aligned_image_path", path: generatedArtifacts.aligned_image_path },
    { source: "artifacts.aligned_image_path", path: artifacts.aligned_image_path },
    { source: "artifacts.original_image_path", path: artifacts.original_image_path },
  ];

  for (const candidate of candidates) {
    const normalizedPath =
      typeof candidate.path === "string" && candidate.path.trim()
        ? candidate.path.trim()
        : null;
    const url = artifactUrl(normalizedPath, cacheKey, buildArtifactUrl);
    if (normalizedPath && url) {
      return {
        source: candidate.source,
        path: normalizedPath,
        url,
      };
    }
  }

  return {
    source: "none",
    path: null,
    url: null,
  };
}
