<template>
  <div class="score-card">
    <div v-if="errorMessage" class="alert alert--warning">
      <strong>{{ text.scoreUnavailableTitle }}</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <template v-else-if="score">
      <div class="score-card__hero">
        <div class="score-card__hero-main">
          <p class="score-card__label">
            <MetricLabel :label="text.scoreLabel" help-key="primary_score" :locale="locale" />
          </p>
          <div class="score-card__value-row">
            <strong class="score-card__value">{{ formatNumber(displayedPrimaryScore, 2) }}</strong>
            <span class="score-card__out-of">{{ scoreRangeLabel }}</span>
          </div>
          <p class="score-card__caption">{{ text.scorePrimaryCaption }}</p>
        </div>
        <div class="score-card__hero-side">
          <span class="score-card__confidence" :class="`score-card__confidence--${confidenceLevel}`">
            <MetricLabel :label="confidenceLabel" help-key="confidence" :locale="locale" />
          </span>
          <p class="score-card__model">
            <MetricLabel :label="text.scoreModelLabel" help-key="primary_score" :locale="locale" />
            <strong>{{ primaryScore.model ?? "regression" }}</strong>
          </p>
        </div>
      </div>

      <div class="score-card__bar">
        <div class="score-card__bar-fill" :style="{ width: `${scorePercent}%` }"></div>
      </div>

      <KeyValueGrid
        :locale="locale"
        :value="{
          [text.scoreFeatureSetLabel]: primaryScore.feature_set_name,
          [text.scoreActiveSourceLabel]: scoreSourcePolicyLabel,
          [text.scoreDirectMediapipeUsedLabel]: primaryScore.direct_mediapipe_features_used ?? primaryScore.mediapipe_features_used,
          [text.scoreMediapipeDerivedRefUsedLabel]: primaryScore.mediapipe_derived_reference_features_used,
          [text.scoreMediapipeUsedLabel]: primaryScore.total_mediapipe_influence_count ?? primaryScore.mediapipe_features_used,
          [text.scoreNearestGroupLabel]: primaryScore.nearest_score_group,
          [text.scoreDistanceToGroupLabel]: primaryScore.distance_to_nearest_group,
          [text.scoreRangeLabel]: scoreRangeText,
          [text.scoreObservedRangeLabel]: observedRangeText,
          [text.scoreSourceLabel]: primaryScore.score_source,
          [text.scoreRawValueLabel]: showRawScore ? primaryScore.raw_predicted_score : null,
          [text.scoreCalibratedValueLabel]: showCalibratedScore ? primaryScore.calibrated_predicted_score : null,
        }"
      />
      <p class="score-card__muted">{{ text.scoreNearestGroupDisclaimer }}</p>
      <p v-if="showCalibratedScore" class="score-card__muted">{{ text.scoreCalibratedDiagnosticNote }}</p>
      <p
        v-if="(primaryScore.direct_mediapipe_features_used ?? primaryScore.mediapipe_features_used ?? 0) === 0
          && (primaryScore.mediapipe_derived_reference_features_used ?? 0) > 0"
        class="score-card__muted"
      >
        {{ text.scoreMediapipeDerivedOnlyNote }}
      </p>

      <p class="score-card__disclaimer">{{ text.scoreExperimentalDisclaimer }}</p>
      <p
        v-if="primaryScore.expanded_geometry_available && !primaryScore.expanded_geometry_used_in_primary"
        class="score-card__muted"
      >
        {{ text.scoreMediapipeNotUsedPrimary }}
      </p>

      <div v-if="confidenceReasons.length" class="alert" :class="confidenceAlertClass">
        <strong>{{ confidenceLabel }}</strong>
        <ul class="score-card__warnings">
          <li v-for="reason in confidenceReasons" :key="reason">{{ reason }}</li>
        </ul>
      </div>

      <div v-if="score.warnings?.length" class="alert alert--warning">
        <strong>{{ text.scoreWarningsTitle }}</strong>
        <ul class="score-card__warnings">
          <li v-for="warning in score.warnings" :key="warning">{{ warning }}</li>
        </ul>
      </div>

      <section class="score-card__section">
        <h4 class="score-card__section-title">
          <MetricLabel :label="text.scoreExplanationTitle" help-key="top_positive_feature" :locale="locale" />
        </h4>
        <template v-if="localExplanation">
          <KeyValueGrid
            :locale="locale"
            :value="{
              [text.scoreExplainedValueLabel]: localExplanation.predicted_score ?? primaryScore.predicted_score,
              [text.scoreExplanationMethodLabel]: localExplanation.explanation_method,
            }"
          />
          <div class="score-card__feature-columns">
            <div class="score-card__feature-column">
              <h5>
                <MetricLabel :label="text.scoreUpwardContribution" help-key="top_positive_feature" :locale="locale" />
              </h5>
              <ul class="score-card__feature-list">
                <li v-for="item in topPositiveFeatures" :key="`pos-${item.feature}`">
                  <span>
                    {{ item.readable_label || item.feature }}
                    <small v-if="item.anatomical_component || item.component" class="score-card__chip">{{ item.anatomical_component || item.component }}</small>
                    <small v-if="item.transform_family" class="score-card__chip">{{ item.transform_family }}</small>
                    <small v-if="typeof item.global_weight === 'number'" class="score-card__chip">w={{ formatNumber(item.global_weight, 2) }}</small>
                  </span>
                  <strong>+{{ formatNumber(item.contribution, 2) }}</strong>
                </li>
              </ul>
            </div>
            <div class="score-card__feature-column">
              <h5>
                <MetricLabel :label="text.scoreDownwardContribution" help-key="top_negative_feature" :locale="locale" />
              </h5>
              <ul class="score-card__feature-list">
                <li v-for="item in topNegativeFeatures" :key="`neg-${item.feature}`">
                  <span>
                    {{ item.readable_label || item.feature }}
                    <small v-if="item.anatomical_component || item.component" class="score-card__chip">{{ item.anatomical_component || item.component }}</small>
                    <small v-if="item.transform_family" class="score-card__chip">{{ item.transform_family }}</small>
                    <small v-if="typeof item.global_weight === 'number'" class="score-card__chip">w={{ formatNumber(item.global_weight, 2) }}</small>
                  </span>
                  <strong>{{ formatNumber(item.contribution, 2) }}</strong>
                </li>
              </ul>
            </div>
          </div>
        </template>
        <p v-else class="score-card__muted">{{ text.scoreExplanationUnavailable }}</p>
      </section>

      <section class="score-card__section">
        <h4 class="score-card__section-title">
          <MetricLabel :label="text.scoreComponentsTitle" help-key="component_contribution" :locale="locale" />
        </h4>
        <template v-if="componentEntries.length">
          <div class="score-card__components">
            <article v-for="[name, component] in componentEntries" :key="name" class="score-card__component">
              <div class="score-card__component-head">
                <div>
                  <strong>{{ name }}</strong>
                  <p class="score-card__component-caption">
                    {{ componentDirectionText(component.direction) }}
                  </p>
                </div>
                <span class="score-card__component-value" :class="`score-card__component-value--${component.direction}`">
                  {{ formatSigned(component.contribution) }}
                </span>
              </div>
              <ul class="score-card__feature-list">
                <li v-for="item in component.top_features ?? []" :key="`${name}-${item.feature}`">
                  <span>{{ item.feature }}</span>
                  <strong>{{ formatSigned(item.contribution) }}</strong>
                </li>
              </ul>
            </article>
          </div>
        </template>
        <p v-else class="score-card__muted">{{ text.scoreExplanationUnavailable }}</p>
      </section>

      <section class="score-card__section">
        <h4 class="score-card__section-title">{{ text.scoreWeightsTitle }}</h4>
        <p class="score-card__muted">{{ text.scoreWeightsDisclaimer }}</p>
        <template v-if="anatomicalComponentWeights.length">
          <ul class="score-card__feature-list">
            <li v-for="item in anatomicalComponentWeights" :key="`weight-${item.component}`">
              <span>{{ item.component }}</span>
              <strong>{{ formatNumber((Number(item.normalized_weight) || 0) * 100, 1) }}%</strong>
            </li>
          </ul>
        </template>
        <p v-else class="score-card__muted">{{ text.scoreExplanationUnavailable }}</p>
        <template v-if="transformFamilyWeights.length">
          <h5>{{ text.scoreFeatureTypesTitle }}</h5>
          <ul class="score-card__feature-list">
            <li v-for="item in transformFamilyWeights" :key="`transform-${item.transform_family}`">
              <span>{{ item.transform_family }}</span>
              <strong>{{ formatNumber((Number(item.normalized_weight) || 0) * 100, 1) }}%</strong>
            </li>
          </ul>
        </template>
        <template v-if="sourceWeights.length">
          <h5>{{ text.scoreFeatureSourcesTitle }}</h5>
          <ul class="score-card__feature-list">
            <li v-for="item in sourceWeights" :key="`source-${item.source_family}`">
              <span>{{ item.source_family }}</span>
              <strong>{{ formatNumber((Number(item.normalized_weight) || 0) * 100, 1) }}%</strong>
            </li>
          </ul>
        </template>
      </section>

      <details class="score-card__details">
        <summary>
          <MetricLabel :label="text.scoreDiagnosticTitle" help-key="auxiliary_diagnostics" :locale="locale" />
        </summary>
        <div class="score-card__details-body">
          <p class="score-card__muted">{{ text.scoreDiagnosticsCaption }}</p>
          <div
            v-if="classifierDisagrees"
            class="alert alert--warning"
          >
            <strong>Classifier diagnostic disagrees with primary score.</strong>
          </div>
          <div v-if="diagnosticDisagreement.length" class="alert alert--info">
            <strong>{{ text.scoreDiagnosticDisagreementTitle }}</strong>
            <ul class="score-card__warnings">
              <li v-for="item in diagnosticDisagreement" :key="item">{{ item }}</li>
            </ul>
          </div>
          <KeyValueGrid
            :locale="locale"
            :value="{
              pairwise_score: auxiliaryScores.pairwise?.approx_score,
              ordinal_score: auxiliaryScores.ordinal?.predicted_score,
              classifier_predicted_group: auxiliaryScores.classifier?.predicted_group,
              ensemble_score: auxiliaryScores.ensemble?.predicted_score,
            }"
          />
          <KeyValueGrid :locale="locale" :value="auxiliaryScores.classifier?.group_probabilities ?? {}" />
          <KeyValueGrid :locale="locale" :value="generatedArtifacts" />
        </div>
      </details>

      <details class="score-card__details">
        <summary>{{ text.scoreDebugTitle }}</summary>
        <div class="score-card__details-body">
          <KeyValueGrid
            :locale="locale"
            :value="{
              [text.scoreExplanationBaselineLabel]: localExplanation?.baseline_score,
              technical_intercept_note: localExplanation?.baseline_score != null ? (locale === 'ru' ? 'Технический intercept модели (debug-only), не отдельная оценка лица.' : 'Technical model intercept (debug-only), not a standalone face score.') : null,
              explained_score: localExplanation?.predicted_score ?? null,
              primary_predicted_score: primaryScore.predicted_score ?? null,
              [text.scoreFeatureSetLabel]: scoringModelInfo.primary_feature_set ?? primaryScore.feature_set_name,
              [text.scoreSelectedFeatureCount]: scoringModelInfo.selected_feature_count,
              [text.scoreMediapipeUsedLabel]: primaryScore.mediapipe_features_used,
              [text.scoreReferenceUsedCount]: primaryScore.reference_relative_features_used,
            }"
          />
          <KeyValueGrid :locale="locale" :value="scoringModelInfo.selected_component_counts ?? {}" />
          <ul class="score-card__feature-list">
            <li v-for="row in topFeatureWeights" :key="`global-weight-${String(row.feature_name)}`">
              <span>{{ String(row.feature_name) }}</span>
              <strong>{{ formatNumber(Number(row.normalized_weight ?? 0), 3) }}</strong>
            </li>
          </ul>
        </div>
      </details>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type {
  AuxiliaryScores,
  ComponentContribution,
  FeatureContribution,
  LocalExplanation,
  PrimaryScore,
  ScoreResponse,
  ScoringModelInfo,
} from "../types/api";
import { formatNumber } from "../utils/format";
import type { Locale } from "../utils/i18n";
import { messages } from "../utils/i18n";
import KeyValueGrid from "./KeyValueGrid.vue";
import MetricLabel from "./MetricLabel.vue";

const props = defineProps<{
  score: ScoreResponse | null;
  errorMessage?: string | null;
  locale: Locale;
}>();

const text = computed(() => messages[props.locale]);
const primaryScore = computed<PrimaryScore>(() => props.score?.primary_score ?? props.score?.regression ?? { predicted_score: 0 });
const auxiliaryScores = computed<AuxiliaryScores>(() => {
  if (props.score?.auxiliary_scores) {
    return props.score.auxiliary_scores;
  }
  return {
    regression: props.score?.regression,
    pairwise: props.score?.pairwise,
    ordinal: props.score?.ordinal,
    classifier: props.score?.classification,
    ensemble: props.score?.final_score
      ? {
          predicted_score: props.score.final_score.predicted_score,
          nearest_score_group: props.score.final_score.nearest_score_group,
          std: props.score.final_score.model_agreement?.std ?? null,
        }
      : null,
  };
});
const localExplanation = computed<LocalExplanation | null>(() => props.score?.local_explanation ?? null);
const topPositiveFeatures = computed<FeatureContribution[]>(() => localExplanation.value?.top_positive_features ?? []);
const topNegativeFeatures = computed<FeatureContribution[]>(() => localExplanation.value?.top_negative_features ?? []);
const componentEntries = computed<Array<[string, ComponentContribution]>>(() =>
  Object.entries(props.score?.component_contributions ?? {}),
);
const confidenceLevel = computed(() => props.score?.confidence.level ?? "unknown");
const confidenceReasons = computed(() => props.score?.confidence.reasons ?? []);
const confidenceLabel = computed(() => {
  const level = confidenceLevel.value;
  if (level === "high") return text.value.scoreConfidenceHigh;
  if (level === "medium") return text.value.scoreConfidenceMedium;
  if (level === "low") return text.value.scoreConfidenceLow;
  return level;
});
const confidenceAlertClass = computed(() => (confidenceLevel.value === "low" ? "alert--warning" : "alert--info"));
const displayedPrimaryScore = computed(() => primaryScore.value.display_score ?? primaryScore.value.predicted_score ?? 0);
const scoreRangeText = computed(() => {
  const [lower, upper] = primaryScore.value.score_range ?? [];
  if (typeof lower === "number" && typeof upper === "number") {
    return `${formatNumber(lower, 0)}-${formatNumber(upper, 0)}`;
  }
  return "0-10";
});
const observedRangeText = computed(() => {
  const [lower, upper] = primaryScore.value.observed_training_label_range ?? [];
  if (typeof lower === "number" && typeof upper === "number") {
    return `${formatNumber(lower, 1)}-${formatNumber(upper, 1)}`;
  }
  return "—";
});
const scoreRangeLabel = computed(() => {
  const parts = scoreRangeText.value.split("-");
  return `/ ${parts[parts.length - 1] ?? "10"}`;
});
const scorePercent = computed(() => {
  const [lowerRaw, upperRaw] = primaryScore.value.score_range ?? [];
  const value = primaryScore.value.predicted_score ?? 0;
  const lower = typeof lowerRaw === "number" ? lowerRaw : 0;
  const upper = typeof upperRaw === "number" ? upperRaw : 10;
  const span = Math.max(upper - lower, 1e-6);
  return Math.max(0, Math.min(100, ((value - lower) / span) * 100));
});
const generatedArtifacts = computed(() => props.score?.generated_artifacts ?? {});
const scoringModelInfo = computed<ScoringModelInfo>(() => props.score?.scoring_model_info ?? {});
const scoreSourcePolicyLabel = computed(() => {
  const policy = (primaryScore.value as unknown as { active_feature_source_policy?: { mode?: string } }).active_feature_source_policy;
  const infoPolicy = (scoringModelInfo.value as unknown as { active_feature_source_policy?: { mode?: string } }).active_feature_source_policy;
  const mode = String(policy?.mode ?? infoPolicy?.mode ?? "");
  if (mode === "mediapipe_only") return "MediaPipe FaceLandmarker dense landmarks";
  return mode || "mixed";
});
const learnedComponentWeights = computed<Array<Record<string, unknown>>>(() => props.score?.learned_weights?.component_weights ?? []);
const anatomicalComponentWeights = computed<Array<Record<string, unknown>>>(() =>
  props.score?.learned_weights?.anatomical_component_weights?.length
    ? props.score?.learned_weights?.anatomical_component_weights ?? []
    : learnedComponentWeights.value.filter((item) => String(item.component ?? "") !== "reference_deviation"),
);
const transformFamilyWeights = computed<Array<Record<string, unknown>>>(() => props.score?.learned_weights?.transform_family_weights ?? []);
const sourceWeights = computed<Array<Record<string, unknown>>>(() => props.score?.learned_weights?.source_weights ?? []);
const topFeatureWeights = computed<Array<Record<string, unknown>>>(() => props.score?.learned_weights?.top_feature_weights ?? []);
const showRawScore = computed(() =>
  typeof primaryScore.value.raw_predicted_score === "number",
);
const showCalibratedScore = computed(() =>
  typeof primaryScore.value.calibrated_predicted_score === "number" &&
  typeof primaryScore.value.calibration_method === "string",
);
const classifierDisagrees = computed(() => {
  const primaryGroup = primaryScore.value.nearest_score_group;
  const classifierGroup = auxiliaryScores.value.classifier?.predicted_group;
  if (primaryGroup == null || !classifierGroup) {
    return false;
  }
  return String(primaryGroup) !== classifierGroup;
});
const diagnosticDisagreement = computed(() => {
  const payload = props.score?.diagnostic_disagreement ?? {};
  const rows: string[] = [];
  const p2 = payload.primary_vs_pairwise_delta;
  const o2 = payload.primary_vs_ordinal_delta;
  const c2 = payload.primary_vs_classifier_group_delta;
  if (typeof p2 === "number") rows.push(`primary vs pairwise: ${formatNumber(p2, 2)}`);
  if (typeof o2 === "number") rows.push(`primary vs ordinal: ${formatNumber(o2, 2)}`);
  if (typeof c2 === "number") rows.push(`primary vs classifier group: ${formatNumber(c2, 2)}`);
  if (payload.pairwise_calibration_reliability === "low") rows.push("pairwise reliability: low (ranking-only diagnostic)");
  return rows;
});

function componentDirectionText(direction: string): string {
  if (direction === "positive") {
    return text.value.scoreUpwardContribution;
  }
  if (direction === "negative") {
    return text.value.scoreDownwardContribution;
  }
  return text.value.scoreModelContribution;
}

function formatSigned(value: number | null | undefined): string {
  if (typeof value !== "number") {
    return "—";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}`;
}
</script>

<style scoped>
.score-card {
  display: grid;
  gap: 1rem;
}

.score-card__hero {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
}

.score-card__hero-main,
.score-card__hero-side {
  display: grid;
  gap: 0.45rem;
}

.score-card__hero-side {
  justify-items: end;
}

.score-card__label,
.score-card__caption,
.score-card__muted,
.score-card__disclaimer,
.score-card__component-caption,
.score-card__model {
  margin: 0;
  color: var(--color-ink-soft);
}

.score-card__value-row {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
}

.score-card__value {
  font-size: clamp(2rem, 5vw, 3.2rem);
  line-height: 1;
}

.score-card__out-of {
  color: var(--color-ink-soft);
}

.score-card__confidence {
  border-radius: 999px;
  padding: 0.45rem 0.8rem;
  font-size: 0.85rem;
  font-weight: 600;
  background: rgba(33, 70, 56, 0.08);
}

.score-card__confidence--high {
  background: rgba(33, 70, 56, 0.12);
  color: #214638;
}

.score-card__confidence--medium {
  background: rgba(179, 126, 26, 0.14);
  color: #7a5512;
}

.score-card__confidence--low {
  background: rgba(168, 62, 40, 0.14);
  color: #8d301d;
}

.score-card__bar {
  height: 0.7rem;
  border-radius: 999px;
  background: rgba(16, 46, 79, 0.1);
  overflow: hidden;
}

.score-card__bar-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #b56b2d, #dfb457 48%, #2d7a58);
}

.score-card__warnings,
.score-card__feature-list {
  margin: 0.5rem 0 0;
  padding-left: 1rem;
}

.score-card__chip {
  margin-left: 0.35rem;
  padding: 0.05rem 0.35rem;
  border-radius: 999px;
  background: rgba(16, 46, 79, 0.08);
  color: var(--color-ink-soft);
  font-size: 0.75rem;
}

.score-card__feature-columns,
.score-card__components {
  display: grid;
  gap: 1rem;
}

.score-card__feature-columns {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.score-card__component {
  border: 1px solid var(--color-line);
  border-radius: 1rem;
  padding: 0.9rem 1rem;
  background: rgba(255, 255, 255, 0.55);
}

.score-card__component-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
}

.score-card__component-value {
  font-weight: 700;
}

.score-card__component-value--positive {
  color: #215c44;
}

.score-card__component-value--negative {
  color: #8d301d;
}

.score-card__component-value--neutral {
  color: var(--color-ink-soft);
}

.score-card__section {
  display: grid;
  gap: 0.8rem;
}

.score-card__section-title {
  margin: 0;
}

.score-card__details summary {
  cursor: pointer;
  font-weight: 600;
}

.score-card__details-body {
  display: grid;
  gap: 1rem;
  margin-top: 0.9rem;
}

@media (max-width: 720px) {
  .score-card__hero {
    grid-template-columns: 1fr;
    display: grid;
  }

  .score-card__hero-side {
    justify-items: start;
  }
}
</style>
