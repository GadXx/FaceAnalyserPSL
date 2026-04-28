import type { AnalyzeResponse } from "../types/api";

export interface SelectedResultImage {
  source: string;
  path: string | null;
  url: string | null;
}

export function selectResultImage(
  result: AnalyzeResponse | null,
  cacheKey: number | null | undefined,
  buildArtifactUrl: (path: string | null | undefined) => string | null,
): SelectedResultImage;
