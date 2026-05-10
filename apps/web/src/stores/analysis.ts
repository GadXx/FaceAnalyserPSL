import { computed, ref } from "vue";
import { defineStore } from "pinia";

import { analyzeUpload, getHealth, getReferenceGroups } from "../api/client";
import type { AnalyzeResponse, HealthResponse, ReferenceGroupEntry, ReferenceGroupsResponse } from "../types/api";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024;

export const useAnalysisStore = defineStore("analysis", () => {
  const health = ref<HealthResponse | null>(null);
  const referenceGroups = ref<ReferenceGroupsResponse | null>(null);
  const analysisResult = ref<AnalyzeResponse | null>(null);
  const selectedFile = ref<File | null>(null);
  const previewUrl = ref<string | null>(null);
  const lastSuccessfulAnalysisAt = ref<number | null>(null);
  const isLoadingHealth = ref(false);
  const isLoadingGroups = ref(false);
  const isAnalyzing = ref(false);
  const errorMessage = ref<string | null>(null);
  const selectedGroupKey = ref<string>("");
  const manualGroup = ref<string>("");
  const outputDir = ref<string>("");
  const preferredExecutionProvider = ref<string>("CPUExecutionProvider");
  const deviceId = ref<number | null>(0);
  const strictGpu = ref<boolean>(false);

  const availableGroups = computed<ReferenceGroupEntry[]>(() => referenceGroups.value?.groups ?? []);
  const selectedReferenceGroup = computed<ReferenceGroupEntry | null>(() => {
    const selected = selectedGroupKey.value.trim();
    if (!selected) {
      return null;
    }
    return availableGroups.value.find((group) => group.group_key === selected) ?? null;
  });
  const serializedSelectedGroup = computed<string | undefined>(() => {
    const selected = selectedReferenceGroup.value;
    if (!selected) {
      return undefined;
    }
    const entries = Object.entries(selected.group_values ?? {}).filter(([, value]) => String(value).trim());
    if (!entries.length) {
      return undefined;
    }
    return entries.map(([key, value]) => `${key}=${value}`).join(",");
  });
  const resolvedGroup = computed<string | undefined>(() => {
    const manual = manualGroup.value.trim();
    if (manual) {
      return manual;
    }
    return serializedSelectedGroup.value;
  });

  async function loadHealth(): Promise<void> {
    isLoadingHealth.value = true;
    try {
      health.value = await getHealth();
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : "Failed to reach backend health endpoint.";
    } finally {
      isLoadingHealth.value = false;
    }
  }

  async function loadReferenceGroups(): Promise<void> {
    isLoadingGroups.value = true;
    try {
      referenceGroups.value = await getReferenceGroups();
      if (!selectedGroupKey.value && referenceGroups.value.groups.length > 0) {
        selectedGroupKey.value = referenceGroups.value.groups[0].group_key;
      }
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : "Failed to load reference groups.";
    } finally {
      isLoadingGroups.value = false;
    }
  }

  function clearSelectedFile(): void {
    selectedFile.value = null;
    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value);
    }
    previewUrl.value = null;
  }

  function setSelectedFile(file: File | null): void {
    errorMessage.value = null;
    if (!file) {
      clearSelectedFile();
      return;
    }
    if (!ACCEPTED_TYPES.includes(file.type)) {
      throw new Error("Unsupported file type. Please upload JPEG, PNG, WEBP, or BMP.");
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      throw new Error("File is too large. Please upload an image up to 15 MB.");
    }
    clearSelectedFile();
    selectedFile.value = file;
    previewUrl.value = URL.createObjectURL(file);
  }

  async function runAnalysis(): Promise<void> {
    if (!selectedFile.value) {
      throw new Error("Choose an image file before starting the analysis.");
    }
    errorMessage.value = null;
    isAnalyzing.value = true;
    try {
      analysisResult.value = await analyzeUpload(selectedFile.value, {
        group: resolvedGroup.value,
        outputDir: outputDir.value.trim() || undefined,
        preferredExecutionProvider: preferredExecutionProvider.value,
        deviceId: deviceId.value ?? undefined,
        strictGpu: strictGpu.value,
      });
      lastSuccessfulAnalysisAt.value = Date.now();
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : "Analysis request failed.";
      throw error;
    } finally {
      isAnalyzing.value = false;
    }
  }

  return {
    health,
    referenceGroups,
    analysisResult,
    selectedFile,
    previewUrl,
    lastSuccessfulAnalysisAt,
    isLoadingHealth,
    isLoadingGroups,
    isAnalyzing,
    errorMessage,
    selectedGroupKey,
    manualGroup,
    outputDir,
    preferredExecutionProvider,
    deviceId,
    strictGpu,
    availableGroups,
    selectedReferenceGroup,
    serializedSelectedGroup,
    resolvedGroup,
    loadHealth,
    loadReferenceGroups,
    setSelectedFile,
    clearSelectedFile,
    runAnalysis,
  };
});
