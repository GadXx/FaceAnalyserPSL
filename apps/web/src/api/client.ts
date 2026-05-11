import type {
  AnalyzeRequestOptions,
  AnalyzeResponse,
  ApiErrorPayload,
  HealthResponse,
  ReferenceGroupsResponse,
  ScoreResponse,
} from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let payload: ApiErrorPayload | null = null;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = null;
    }
    const nestedDetail =
      payload && typeof payload.detail === "object" && payload.detail
        ? payload.detail.detail
        : undefined;
    const message =
      nestedDetail ||
      (typeof payload?.detail === "string" ? payload.detail : undefined) ||
      `API request failed with status ${response.status}`;
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return parseJsonResponse<HealthResponse>(await fetch(`/health`));
}

export async function getReferenceGroups(): Promise<ReferenceGroupsResponse> {
  return parseJsonResponse<ReferenceGroupsResponse>(await fetch(`/reference-groups`));
}

export async function analyzeUpload(file: File, options: AnalyzeRequestOptions): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.group) {
    formData.append("group", options.group);
  }
  if (options.outputDir) {
    formData.append("output_dir", options.outputDir);
  }
  if (options.preferredExecutionProvider) {
    formData.append("preferred_execution_provider", options.preferredExecutionProvider);
  }
  if (typeof options.deviceId === "number") {
    formData.append("device_id", String(options.deviceId));
  }
  if (typeof options.strictGpu === "boolean") {
    formData.append("strict_gpu", String(options.strictGpu));
  }
  return parseJsonResponse<AnalyzeResponse>(
    await fetch(`/analyze-upload`, {
      method: "POST",
      body: formData,
    }),
  );
}

export async function scoreUpload(file: File): Promise<ScoreResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return parseJsonResponse<ScoreResponse>(
    await fetch(`/score-upload`, {
      method: "POST",
      body: formData,
    }),
  );
}

export function buildArtifactFileUrl(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }
  return `/artifact-file?path=${encodeURIComponent(path)}`;
}

export { API_BASE_URL };
