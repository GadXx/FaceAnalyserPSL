<template>
  <div class="app-shell">
    <aside class="app-shell__sidebar">
      <div class="brand-card">
        <div class="brand-card__topline">
          <p class="brand-card__eyebrow">{{ text.appEyebrow }}</p>
          <label class="language-switcher">
            <span>{{ text.languageLabel }}</span>
            <select v-model="locale">
              <option value="ru">{{ text.languageRu }}</option>
              <option value="en">{{ text.languageEn }}</option>
            </select>
          </label>
        </div>
        <h1>{{ text.appTitle }}</h1>
        <p>{{ text.appDescription }}</p>
      </div>

      <SectionCard :title="text.backendStatusTitle" :eyebrow="text.backendStatusEyebrow">
        <template #header-extra>
          <span class="status-pill" :class="healthStatusClass">{{ healthLabel }}</span>
        </template>
        <div v-if="store.isLoadingHealth" class="inline-loading">{{ text.backendChecking }}</div>
        <div v-else-if="store.health">
          <KeyValueGrid
            :locale="locale"
            :value="{
              [text.statusKey]: store.health.status,
              [text.canAnalyzeKey]: store.health.can_analyze,
              [text.dependencyCountKey]: store.health.dependencies.length,
            }"
          />
          <ul class="dependency-list">
            <li v-for="dependency in store.health.dependencies" :key="dependency.name">
              <span>{{ dependency.name }}</span>
              <strong :class="{ 'text-danger': !dependency.available }">
                {{ dependency.available ? text.healthAvailable : text.healthMissing }}
              </strong>
            </li>
          </ul>
        </div>
      </SectionCard>

      <SectionCard :title="text.uploadTitle" :eyebrow="text.uploadEyebrow" :locale="locale">
        <UploadDropzone
          :error-message="store.errorMessage"
          :has-file="Boolean(store.selectedFile)"
          :title="text.uploadDropTitle"
          :description="text.uploadDropDescription"
          :choose-label="text.uploadChoose"
          :clear-label="text.uploadClear"
          @select="handleFileSelected"
          @clear="store.clearSelectedFile"
        />
        <div v-if="store.selectedFile" class="preview-card">
          <img v-if="store.previewUrl" :src="store.previewUrl" :alt="text.previewAlt" class="preview-card__image" />
          <div class="preview-card__meta">
            <p class="preview-card__name">{{ store.selectedFile.name }}</p>
            <p>{{ formatNumber(store.selectedFile.size / (1024 * 1024), 2) }} MB</p>
            <p>{{ store.selectedFile.type || text.unknownContentType }}</p>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        :title="text.analysisTitle"
        :eyebrow="text.analysisEyebrow"
        :description="text.analysisDescription"
        help-key="section.analysis_request"
        :locale="locale"
      >
        <div class="form-grid">
          <label class="field">
            <MetricLabel :label="text.referenceGroup" help-key="field.reference_group_selector" :locale="locale" />
            <select v-model="store.selectedGroupKey">
              <option value="">{{ text.referenceGroupPlaceholder }}</option>
              <option v-for="group in store.availableGroups" :key="group.group_key" :value="group.group_key">
                {{ group.group_key }} (n={{ group.sample_count }})
              </option>
            </select>
            <small class="field__help">{{ text.referenceGroupHelp }}</small>
          </label>
          <label class="field">
            <MetricLabel :label="text.manualGroup" help-key="field.manual_group_override" :locale="locale" />
            <input
              v-model="store.manualGroup"
              type="text"
              :placeholder="text.manualGroupPlaceholder"
            />
            <small class="field__help">{{ text.manualGroupHelp }}</small>
          </label>
          <label class="field">
            <MetricLabel :label="text.executionProvider" help-key="field.execution_provider" :locale="locale" />
            <select v-model="store.preferredExecutionProvider">
              <option value="CPUExecutionProvider">CPUExecutionProvider</option>
              <option value="CUDAExecutionProvider">CUDAExecutionProvider</option>
            </select>
            <small class="field__help">{{ text.executionProviderHelp }}</small>
          </label>
          <label class="field">
            <MetricLabel :label="text.deviceId" help-key="field.device_id" :locale="locale" />
            <input v-model.number="store.deviceId" type="number" min="0" />
            <small class="field__help">{{ text.deviceIdHelp }}</small>
          </label>
          <label class="field field--wide">
            <MetricLabel :label="text.outputDir" help-key="field.output_dir" :locale="locale" />
            <input v-model="store.outputDir" type="text" placeholder="reports/examples" />
            <small class="field__help">{{ text.outputDirHelp }}</small>
          </label>
          <label class="toggle">
            <input v-model="store.strictGpu" type="checkbox" />
            <MetricLabel :label="text.strictGpu" help-key="field.strict_gpu" :locale="locale" />
          </label>
          <p class="toggle__help">{{ text.strictGpuHelp }}</p>
        </div>
        <div class="section-actions">
          <button class="button button--primary button--wide" type="button" @click="submitAnalysis" :disabled="store.isAnalyzing">
            {{ store.isAnalyzing ? text.analyzing : text.runAnalysis }}
          </button>
        </div>
      </SectionCard>
    </aside>

    <main class="app-shell__main">
      <div v-if="store.errorMessage" class="alert alert--error">
        <strong>{{ text.requestError }}</strong>
        <span>{{ store.errorMessage }}</span>
      </div>

      <LoadingSkeleton v-if="store.isAnalyzing" />

      <template v-else-if="store.analysisResult">
        <SectionCard
          :title="text.contextTitle"
          :eyebrow="text.contextEyebrow"
          :description="text.contextDescription"
          help-key="section.context"
          :locale="locale"
        >
          <KeyValueGrid :value="contextSummary" :locale="locale" />
        </SectionCard>

        <SectionCard
          :title="text.qualityTitle"
          :eyebrow="text.qualityEyebrow"
          :description="text.qualityDescription"
          help-key="section.quality"
          :locale="locale"
        >
          <div class="section-stack">
            <KeyValueGrid
              :locale="locale"
              :value="{
                processing_status: store.analysisResult.input_quality.processing_status,
                quality_pass: store.analysisResult.input_quality.quality_pass,
                quality_fail_reason: store.analysisResult.input_quality.quality_fail_reason,
                processing_error_message: store.analysisResult.input_quality.processing_error_message,
              }"
            />
            <KeyValueGrid
              :locale="locale"
              :value="{
                bbox: store.analysisResult.input_quality.detection.bbox,
                det_score: store.analysisResult.input_quality.detection.det_score,
                yaw: store.analysisResult.input_quality.detection.pose.yaw,
                pitch: store.analysisResult.input_quality.detection.pose.pitch,
                roll: store.analysisResult.input_quality.detection.pose.roll,
                num_faces: store.analysisResult.input_quality.detection.num_faces,
                detection_backend: store.analysisResult.input_quality.detection.detection_backend,
                blur_variance: store.analysisResult.input_quality.detection.blur_variance,
                occlusion_score: store.analysisResult.input_quality.detection.occlusion_score,
              }"
            />
          </div>
        </SectionCard>

        <SectionCard
          :title="text.visualAnalysisTitle"
          :eyebrow="text.visualAnalysisEyebrow"
          :description="text.visualAnalysisDescription"
          help-key="section.visual_analysis"
          :locale="locale"
        >
          <MorphometricOverlay
            :report="store.analysisResult"
            :image-url="analysisImageUrl"
            :locale="locale"
          />
        </SectionCard>

        <div class="result-grid">
          <SectionCard :title="text.rawMeasurementsTitle" :eyebrow="text.geometryEyebrow" help-key="section.raw_measurements" :locale="locale">
            <KeyValueGrid :value="store.analysisResult.raw_measurements" :locale="locale" />
          </SectionCard>

          <SectionCard :title="text.normalizedRatiosTitle" :eyebrow="text.featuresEyebrow" help-key="section.normalized_ratios" :locale="locale">
            <KeyValueGrid :value="store.analysisResult.normalized_ratios" :locale="locale" />
          </SectionCard>

          <SectionCard :title="text.symmetryMetricsTitle" :eyebrow="text.featuresEyebrow" help-key="section.symmetry_metrics" :locale="locale">
            <KeyValueGrid :value="store.analysisResult.symmetry_metrics" :locale="locale" />
          </SectionCard>

          <SectionCard :title="text.angleMetricsTitle" :eyebrow="text.featuresEyebrow" help-key="section.angle_metrics" :locale="locale">
            <KeyValueGrid :value="store.analysisResult.angle_metrics" :locale="locale" />
          </SectionCard>
        </div>

        <SectionCard :title="text.referenceTitle" :eyebrow="text.referenceEyebrow" help-key="section.reference_comparison" :locale="locale">
          <div class="section-stack">
            <KeyValueGrid
              :locale="locale"
              :value="{
                group_key: store.analysisResult.reference_comparison.selected_group.group_key,
                sample_count: store.analysisResult.reference_comparison.selected_group.sample_count,
                complete_case_count: store.analysisResult.reference_comparison.selected_group.complete_case_count,
                mahalanobis_distance: store.analysisResult.reference_comparison.mahalanobis_distance,
              }"
            />
            <KeyValueGrid :value="store.analysisResult.reference_comparison.selected_group.group_values ?? {}" :locale="locale" />
            <KeyValueGrid :value="store.analysisResult.reference_comparison.z_scores" :locale="locale" />
            <KeyValueGrid :value="store.analysisResult.reference_comparison.percentile_positions" :locale="locale" />
          </div>
        </SectionCard>

        <SectionCard
          :title="text.deviationsTitle"
          :eyebrow="text.deviationsEyebrow"
          :description="text.deviationsDescription"
          help-key="section.feature_deviations"
          :locale="locale"
        >
          <DeviationList :items="store.analysisResult.feature_level_deviations.top_contributing_features" :locale="locale" />
        </SectionCard>

        <SectionCard :title="text.compositeTitle" :eyebrow="text.compositeEyebrow" help-key="section.composite_indices" :locale="locale">
          <KeyValueGrid :value="store.analysisResult.optional_composite_indices" :locale="locale" />
        </SectionCard>

        <SectionCard :title="text.limitationsTitle" :eyebrow="text.limitationsEyebrow" help-key="section.limitations" :locale="locale">
          <ul class="limitations-list">
            <li v-for="item in store.analysisResult.limitations" :key="item">{{ item }}</li>
          </ul>
        </SectionCard>

        <SectionCard :title="text.artifactsTitle" :eyebrow="text.artifactsEyebrow" help-key="section.artifacts" :locale="locale">
          <div class="section-stack">
            <KeyValueGrid :value="store.analysisResult.artifacts" :locale="locale" />
            <KeyValueGrid :value="store.analysisResult.generated_artifacts" :locale="locale" />
          </div>
        </SectionCard>
      </template>

      <SectionCard
        v-else
        :title="text.emptyTitle"
        :eyebrow="text.emptyEyebrow"
        :description="text.emptyDescription"
      >
        <p class="empty-state">
          {{ text.emptyBody }}
        </p>
      </SectionCard>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";

import DeviationList from "../components/DeviationList.vue";
import KeyValueGrid from "../components/KeyValueGrid.vue";
import LoadingSkeleton from "../components/LoadingSkeleton.vue";
import MetricLabel from "../components/MetricLabel.vue";
import MorphometricOverlay from "../components/MorphometricOverlay.vue";
import SectionCard from "../components/SectionCard.vue";
import UploadDropzone from "../components/UploadDropzone.vue";
import { buildArtifactFileUrl } from "../api/client";
import { useAnalysisStore } from "../stores/analysis";
import { formatNumber } from "../utils/format";
import { messages, type Locale } from "../utils/i18n";
import { selectResultImage } from "../utils/resultImageSelection";

const store = useAnalysisStore();
const initialLocale = (typeof window !== "undefined" ? window.localStorage.getItem("morphometric-locale") : null) as Locale | null;
const locale = ref<Locale>(initialLocale === "ru" || initialLocale === "en" ? initialLocale : "ru");
const text = computed(() => messages[locale.value]);

const healthLabel = computed(() => {
  if (store.isLoadingHealth) {
    return text.value.healthChecking;
  }
  return store.health?.status ?? text.value.healthUnknown;
});

const healthStatusClass = computed(() => ({
  "status-pill--ok": store.health?.status === "ok",
  "status-pill--warning": store.health?.status === "degraded",
}));

const contextSummary = computed<Record<string, unknown>>(() => {
  const result = store.analysisResult;
  if (!result) {
    return {};
  }
  return {
    [text.value.inputSource]: result.input.image_path,
    [text.value.uploadOriginalFilename]: result.input.upload_original_filename,
    [text.value.uploadContentType]: result.input.upload_content_type,
    [text.value.outputDirectoryUsed]: result.input.output_dir,
    [text.value.referenceGroupUsed]: result.reference_comparison.selected_group.group_key,
    [text.value.executionProviderUsed]: result.input.preferred_execution_provider,
    [text.value.deviceIdUsed]: result.input.device_id,
    [text.value.resultImageSource]: selectedResultImage.value.source,
    [text.value.analysisImagePath]: result.generated_artifacts.analysis_image_path,
    [text.value.normalizedFrameImagePath]: result.generated_artifacts.normalized_frame_image_path,
    [text.value.annotatedImagePath]: result.generated_artifacts.annotated_image_path,
    [text.value.alignedImagePath]: result.generated_artifacts.aligned_image_path,
    [text.value.coordinateSpace]: result.visualization?.coordinate_space,
  };
});

const selectedResultImage = computed(() =>
  selectResultImage(store.analysisResult, store.lastSuccessfulAnalysisAt, buildArtifactFileUrl),
);

const analysisImageUrl = computed<string | null>(() => selectedResultImage.value.url);

watch(locale, (value) => {
  if (typeof window !== "undefined") {
    window.localStorage.setItem("morphometric-locale", value);
  }
});

watch(
  selectedResultImage,
  (value) => {
    if (!store.analysisResult) {
      return;
    }
    console.info("[morphometric-ui] selected analysis image source", {
      source: value.source,
      path: value.path,
      url: value.url,
    });
  },
  { deep: true },
);

onMounted(async () => {
  await Promise.all([store.loadHealth(), store.loadReferenceGroups()]);
});

function handleFileSelected(file: File): void {
  try {
    store.setSelectedFile(file);
  } catch (error) {
    store.errorMessage = error instanceof Error ? error.message : locale.value === "ru" ? "Некорректный файл." : "Invalid file.";
  }
}

async function submitAnalysis(): Promise<void> {
  try {
    await store.runAnalysis();
  } catch {
    // Store already holds the user-facing error message.
  }
}
</script>
