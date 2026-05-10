<template>
  <div class="overlay-panel">
    <div class="overlay-toolbar">
      <label v-for="toggle in toggleList" :key="toggle.key" class="overlay-toggle">
        <input v-model="layers[toggle.key]" type="checkbox" />
        <MetricLabel :label="toggle.label" :help-key="toggle.helpKey" :locale="locale" />
      </label>
    </div>

    <div v-if="qualityWarning" class="overlay-alert overlay-alert--warning">
      <strong>{{ copy.warningTitle }}</strong>
      <span>{{ qualityWarning }}</span>
    </div>
    <div v-if="semanticWarning" class="overlay-alert overlay-alert--warning">
      <strong>{{ copy.semanticWarningTitle }}</strong>
      <span>{{ semanticWarning }}</span>
    </div>

    <div v-if="visualizationNotes.length" class="overlay-notes">
      <p class="overlay-notes__title">{{ copy.notesTitle }}</p>
      <ul>
        <li v-for="note in visualizationNotes" :key="note">{{ note }}</li>
      </ul>
    </div>

    <div v-if="!imageUrl" class="overlay-placeholder">
      <strong>{{ copy.placeholderNoImageTitle }}</strong>
      <p>{{ copy.placeholderNoImageBody }}</p>
    </div>
    <div v-else-if="!imageSize" class="overlay-placeholder">
      <strong>{{ copy.placeholderNoOverlayTitle }}</strong>
      <p>{{ copy.placeholderNoOverlayBody }}</p>
    </div>
    <div v-else class="overlay-stage-wrap">
      <div class="overlay-stage" :style="{ aspectRatio: `${imageSize.width} / ${imageSize.height}` }">
        <img :src="imageUrl" :alt="copy.imageAlt" class="overlay-stage__image" />
        <svg
          class="overlay-stage__svg"
          :viewBox="`0 0 ${imageSize.width} ${imageSize.height}`"
          preserveAspectRatio="xMidYMid meet"
        >
          <g v-if="layers.bbox && bboxRect">
            <rect
              class="overlay-box"
              :x="bboxRect.x"
              :y="bboxRect.y"
              :width="bboxRect.width"
              :height="bboxRect.height"
            >
              <title>{{ copy.bboxLegend }}</title>
            </rect>
          </g>

          <g>
            <template v-for="segment in visibleSegments" :key="segment.id">
              <line
                class="overlay-line"
                :class="segment.className"
                :x1="segment.start.x"
                :y1="segment.start.y"
                :x2="segment.end.x"
                :y2="segment.end.y"
              >
                <title>{{ segment.tooltip }}</title>
              </line>
              <text
                v-if="layers.labels && segment.label"
                class="overlay-label"
                :class="segment.labelClassName"
                :x="segment.labelPoint.x"
                :y="segment.labelPoint.y"
                :style="labelTextStyle"
              >
                {{ segment.label }}
              </text>
            </template>
          </g>

          <g v-if="layers.landmarks">
            <template v-for="point in visibleNamedPoints" :key="point.name">
              <circle
                class="overlay-point"
                :class="point.className"
                :cx="point.point.x"
                :cy="point.point.y"
                :r="overlaySizes.pointRadius"
                :style="pointStyle"
              >
                <title>{{ point.label }}</title>
              </circle>
              <text
                v-if="layers.labels"
                class="overlay-label overlay-label--point"
                :x="point.point.x + overlaySizes.pointLabelOffset"
                :y="point.point.y - overlaySizes.pointLabelOffset"
                :style="labelTextStyle"
              >
                {{ point.label }}
              </text>
            </template>
          </g>

          <g v-if="layers.debugMode && layers.allLandmarks">
            <template v-for="landmark in denseLandmarks" :key="`debug-${landmark.index}`">
              <circle
                class="overlay-point overlay-point--debug"
                :class="`overlay-point--${String(landmark.group ?? 'misc').replace(/_/g, '-')}`"
                :cx="landmark.x"
                :cy="landmark.y"
                :r="overlaySizes.debugPointRadius"
                :style="debugPointStyle"
              >
                <title>{{ `#${landmark.index}${landmark.name ? ` ${landmark.name}` : ""}` }}</title>
              </circle>
              <text
                v-if="layers.landmarkIndices"
                class="overlay-label overlay-label--debug"
                :x="landmark.x + overlaySizes.debugLabelOffset"
                :y="landmark.y - overlaySizes.debugLabelOffset"
                :style="debugLabelTextStyle"
              >
                {{ landmark.index }}
              </text>
            </template>
          </g>
        </svg>
      </div>
    </div>

    <div class="overlay-meta-grid">
      <div class="overlay-legend">
        <p class="overlay-legend__title">{{ copy.legendTitle }}</p>
        <ul>
          <li><span class="overlay-swatch overlay-swatch--bbox"></span>{{ copy.bboxLegend }}</li>
          <li><span class="overlay-swatch overlay-swatch--measurement"></span>{{ copy.measurementLegend }}</li>
          <li><span class="overlay-swatch overlay-swatch--symmetry"></span>{{ copy.symmetryLegend }}</li>
          <li><span class="overlay-swatch overlay-swatch--thirds"></span>{{ copy.thirdsLegend }}</li>
          <li><span class="overlay-swatch overlay-swatch--angle"></span>{{ copy.angleLegend }}</li>
          <li><span class="overlay-swatch overlay-swatch--point"></span>{{ copy.landmarkLegend }}</li>
        </ul>
      </div>

      <div class="overlay-highlights">
        <p class="overlay-highlights__title">{{ copy.highlightTitle }}</p>
        <div v-if="highlightItems.length" class="overlay-highlights__list">
          <article v-for="item in highlightItems" :key="item.featureName" class="overlay-highlights__item">
            <p class="overlay-highlights__name">{{ item.featureName }}</p>
            <p class="overlay-highlights__value">{{ item.value }}</p>
            <p class="overlay-highlights__description">{{ item.description }}</p>
          </article>
        </div>
        <p v-else class="overlay-highlights__empty">{{ copy.highlightEmpty }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive } from "vue";

import type {
  AnalyzeResponse,
  FeatureDeviation,
  LandmarkPointResponse,
  Point2DResponse,
  VisualizationLineResponse,
} from "../types/api";
import { formatLabel, formatNumber } from "../utils/format";
import type { Locale } from "../utils/i18n";
import { pointGroupMatches, segmentLayerVisible } from "../utils/overlayVisibility.js";
import { getMeasurementOverlaySegments, getRawLandmarks, getSemanticPoints } from "../utils/visualizationPayload.js";
import MetricLabel from "./MetricLabel.vue";

type ToggleKey =
  | "bbox"
  | "landmarks"
  | "eyes"
  | "nose"
  | "mouth"
  | "chinJaw"
  | "symmetry"
  | "thirds"
  | "angles"
  | "measurements"
  | "debugMode"
  | "allLandmarks"
  | "landmarkIndices"
  | "labels";

type OverlaySegment = {
  id: string;
  start: Point2DResponse;
  end: Point2DResponse;
  label: string | null;
  tooltip: string;
  labelPoint: Point2DResponse;
  className: string;
  labelClassName: string;
  supportedFeatures: string[];
};

type NamedPointEntry = {
  name: string;
  point: Point2DResponse;
  label: string;
  className: string;
  group: string;
};

type HighlightItem = {
  featureName: string;
  value: string;
  description: string;
  primitiveIds: string[];
};

const props = defineProps<{
  report: AnalyzeResponse | null;
  imageUrl: string | null;
  locale: Locale;
}>();

const layers = reactive<Record<ToggleKey, boolean>>({
  bbox: true,
  landmarks: true,
  eyes: true,
  nose: true,
  mouth: true,
  chinJaw: true,
  symmetry: true,
  thirds: true,
  angles: true,
  measurements: true,
  debugMode: false,
  allLandmarks: false,
  landmarkIndices: false,
  labels: false,
});

const overlayMessages = {
  en: {
    toggleBbox: "BBox",
    toggleLandmarks: "Landmarks",
    toggleEyes: "Eyes",
    toggleNose: "Nose",
    toggleMouth: "Mouth",
    toggleChinJaw: "Chin / jaw",
    toggleSymmetry: "Symmetry",
    toggleThirds: "Thirds",
    toggleAngles: "Angles",
    toggleMeasurements: "Measurements",
    toggleDebugMode: "Debug mode",
    toggleAllLandmarks: "All landmarks",
    toggleLandmarkIndices: "Landmark indices",
    toggleLabels: "Labels",
    placeholderNoImageTitle: "Upload an image to see the overlay",
    placeholderNoImageBody: "The visual analysis panel uses the real uploaded image together with preprocessing coordinates from the morphometric API.",
    placeholderNoOverlayTitle: "Overlay data is not available",
    placeholderNoOverlayBody: "The current report does not contain enough real coordinates to draw the morphometric overlay.",
    warningTitle: "Interpretation warning.",
    semanticWarningTitle: "Landmark quality warning.",
    notesTitle: "Overlay notes",
    imageAlt: "Morphometric overlay preview",
    legendTitle: "Legend",
    bboxLegend: "Detected face bounding box",
    measurementLegend: "Primary distances and width / height segments",
    symmetryLegend: "Midline and left/right asymmetry distances",
    thirdsLegend: "Vertical thirds and proportion guides",
    angleLegend: "Axes and angular direction lines",
    landmarkLegend: "Named anchor and derived points",
    highlightTitle: "Top visualizable deviations",
    highlightEmpty: "Current top deviations are either non-visualizable from 2D coordinates or not available.",
  },
  ru: {
    toggleBbox: "BBox",
    toggleLandmarks: "Landmarks",
    toggleEyes: "Глаза",
    toggleNose: "Нос",
    toggleMouth: "Рот",
    toggleChinJaw: "Подбородок / челюсть",
    toggleSymmetry: "Симметрия",
    toggleThirds: "Трети",
    toggleAngles: "Углы",
    toggleMeasurements: "Измерения",
    toggleDebugMode: "Debug режим",
    toggleAllLandmarks: "Все landmarks",
    toggleLandmarkIndices: "Индексы landmarks",
    toggleLabels: "Подписи",
    placeholderNoImageTitle: "Загрузите изображение, чтобы увидеть overlay",
    placeholderNoImageBody: "Панель визуального анализа использует реальное загруженное фото и координаты preprocessing из morphometric API.",
    placeholderNoOverlayTitle: "Недостаточно данных для overlay",
    placeholderNoOverlayBody: "В текущем отчёте не хватает реальных координат, чтобы построить морфометрическое наложение.",
    warningTitle: "Предупреждение об интерпретации.",
    semanticWarningTitle: "Предупреждение по landmark-ам.",
    notesTitle: "Примечания overlay",
    imageAlt: "Предпросмотр морфометрического overlay",
    legendTitle: "Легенда",
    bboxLegend: "Рамка обнаруженного лица",
    measurementLegend: "Основные расстояния и ширины / высоты сегментов",
    symmetryLegend: "Midline и линии лево/правой асимметрии",
    thirdsLegend: "Вертикальные трети и направляющие пропорций",
    angleLegend: "Оси и угловые направляющие линии",
    landmarkLegend: "Именованные опорные и производные точки",
    highlightTitle: "Главные визуализируемые отклонения",
    highlightEmpty: "Текущие top deviations либо не визуализируются по 2D координатам, либо отсутствуют.",
  },
} satisfies Record<Locale, Record<string, string>>;

const copy = computed(() => overlayMessages[props.locale]);
const visualization = computed(() => props.report?.visualization ?? null);
const imageSize = computed(() => visualization.value?.image_size ?? null);
const bbox = computed<number[] | null>(() => visualization.value?.bbox ?? props.report?.input_quality.detection.bbox ?? null);
const visualizationNotes = computed(() => visualization.value?.notes ?? []);
const denseLandmarks = computed<LandmarkPointResponse[]>(
  () => getRawLandmarks(visualization.value),
);
const overlayScale = computed(() => {
  const size = imageSize.value;
  if (!size) {
    return 1;
  }
  const minDimension = Math.min(size.width, size.height);
  return Math.max(0.7, Math.min(0.95, minDimension / 320));
});
const overlaySizes = computed(() => ({
  pointRadius: Number((3.1 * overlayScale.value).toFixed(2)),
  debugPointRadius: Number((1.65 * overlayScale.value).toFixed(2)),
  pointStrokeWidth: Number((1.2 * overlayScale.value).toFixed(2)),
  debugPointStrokeWidth: Number((0.75 * overlayScale.value).toFixed(2)),
  labelFontSize: Number((4.25 * overlayScale.value).toFixed(2)),
  debugLabelFontSize: Number((3.45 * overlayScale.value).toFixed(2)),
  labelStrokeWidth: Number((1.85 * overlayScale.value).toFixed(2)),
  pointLabelOffset: Number((5.5 * overlayScale.value).toFixed(2)),
  debugLabelOffset: Number((2.9 * overlayScale.value).toFixed(2)),
  lineLabelOffset: Number((5.0 * overlayScale.value).toFixed(2)),
  guideLabelOffset: Number((9.0 * overlayScale.value).toFixed(2)),
  angleLabelOffset: Number((10.0 * overlayScale.value).toFixed(2)),
}));
const pointStyle = computed(() => ({
  strokeWidth: overlaySizes.value.pointStrokeWidth,
}));
const debugPointStyle = computed(() => ({
  strokeWidth: overlaySizes.value.debugPointStrokeWidth,
}));
const labelTextStyle = computed(() => ({
  fontSize: `${overlaySizes.value.labelFontSize}px`,
  strokeWidth: `${overlaySizes.value.labelStrokeWidth}px`,
}));
const debugLabelTextStyle = computed(() => ({
  fontSize: `${overlaySizes.value.debugLabelFontSize}px`,
  strokeWidth: `${Math.max(1, overlaySizes.value.labelStrokeWidth * 0.8)}px`,
}));

const toggleList = computed(() => [
  { key: "bbox" as const, label: copy.value.toggleBbox, helpKey: "toggle.bbox" },
  { key: "landmarks" as const, label: copy.value.toggleLandmarks, helpKey: "toggle.landmarks" },
  { key: "eyes" as const, label: copy.value.toggleEyes, helpKey: "toggle.eyes" },
  { key: "nose" as const, label: copy.value.toggleNose, helpKey: "toggle.nose" },
  { key: "mouth" as const, label: copy.value.toggleMouth, helpKey: "toggle.mouth" },
  { key: "chinJaw" as const, label: copy.value.toggleChinJaw, helpKey: "toggle.chin_jaw" },
  { key: "symmetry" as const, label: copy.value.toggleSymmetry, helpKey: "toggle.symmetry" },
  { key: "thirds" as const, label: copy.value.toggleThirds, helpKey: "toggle.thirds" },
  { key: "angles" as const, label: copy.value.toggleAngles, helpKey: "toggle.angles" },
  { key: "measurements" as const, label: copy.value.toggleMeasurements, helpKey: "toggle.raw_measurements" },
  { key: "debugMode" as const, label: copy.value.toggleDebugMode, helpKey: "toggle.debug_mode" },
  { key: "allLandmarks" as const, label: copy.value.toggleAllLandmarks, helpKey: "toggle.all_landmarks" },
  { key: "landmarkIndices" as const, label: copy.value.toggleLandmarkIndices, helpKey: "toggle.debug_indices" },
  { key: "labels" as const, label: copy.value.toggleLabels, helpKey: "toggle.labels" },
]);

const qualityWarning = computed(() => {
  const quality = props.report?.input_quality;
  if (!quality || quality.quality_pass) {
    return "";
  }
  return quality.quality_fail_reason
    ? `${quality.quality_fail_reason}. ${props.locale === "ru" ? "Измерения могут быть менее надёжными." : "Measurements may be less reliable."}`
    : props.locale === "ru"
      ? "Quality gate не пройден; интерпретация может быть менее надёжной."
      : "Quality gate did not pass; interpretation may be less reliable.";
});

const semanticWarning = computed(() => {
  const validation = visualization.value?.semantic_validation;
  if (!validation || validation.passed !== false) {
    return "";
  }
  if (validation.failed_checks?.length) {
    return validation.failed_checks.join(", ");
  }
  return props.locale === "ru"
    ? "Проверка semantic landmark-ов не пройдена; зависимые измерения помечены как ненадёжные."
    : "Semantic landmark validation failed; dependent measurements are marked as unreliable.";
});

const allPoints = computed<Record<string, Point2DResponse>>(() => (
  getSemanticPoints(visualization.value)
));

const pointGroups = computed<Record<string, string>>(() => {
  const groups = visualization.value?.groups ?? {};
  const result: Record<string, string> = {};
  for (const [groupName, members] of Object.entries(groups)) {
    for (const member of members) {
      result[member] = groupName;
    }
  }
  return result;
});

function pointDisplayName(name: string): string {
  const englishMap: Record<string, string> = {
    left_eye_center: "Left eye",
    right_eye_center: "Right eye",
    nose_tip: "Nose tip",
    nose_base_center: "Nose base",
    mouth_left: "Left mouth",
    mouth_right: "Right mouth",
    mouth_center: "Mouth center",
    chin_center: "Chin center",
    brow_left: "Left brow",
    brow_right: "Right brow",
    brow_center: "Brow center",
    jaw_left: "Left jaw",
    jaw_right: "Right jaw",
    cheekbone_left: "Left cheekbone",
    cheekbone_right: "Right cheekbone",
  };
  return props.locale === "ru" ? formatLabel(name) : (englishMap[name] ?? formatLabel(name));
}

const visibleNamedPoints = computed<NamedPointEntry[]>(() => {
  const names = Object.keys(
    visualization.value?.semantic_points ?? {},
  );
  return names
    .map((name) => {
      const point = allPoints.value[name];
      if (!point) {
        return null;
      }
      const group = pointGroups.value[name] ?? "misc";
      if (!pointGroupMatches(group, layers)) {
        return null;
      }
      return {
        name,
        point,
        label: pointDisplayName(name),
        className: `overlay-point--${group.replace(/_/g, "-")}`,
        group,
      };
    })
    .filter((item): item is NamedPointEntry => Boolean(item));
});

const bboxRect = computed(() => {
  if (!bbox.value || bbox.value.length !== 4) {
    return null;
  }
  const [x1, y1, x2, y2] = bbox.value;
  return { x: x1, y: y1, width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1) };
});

function midpoint(start: Point2DResponse, end: Point2DResponse, offsetY = -6): Point2DResponse {
  return { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 + offsetY };
}

function resolveMetricValue(metricKey: string | null | undefined): number | null {
  if (!metricKey || !props.report) {
    return null;
  }
  const report = props.report;
  const maps = [
    report.raw_measurements,
    report.normalized_ratios,
    report.symmetry_metrics,
    report.angle_metrics,
    report.optional_composite_indices,
    report.reference_comparison.z_scores,
    report.reference_comparison.percentile_positions,
  ];
  for (const mapping of maps) {
    const value = mapping?.[metricKey];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function metricLabel(metricKey: string | null | undefined, kind: string | null | undefined): string | null {
  const value = resolveMetricValue(metricKey);
  if (value == null) {
    return null;
  }
  if ((metricKey ?? "").includes("angle") || (metricKey ?? "").endsWith("_deg")) {
    return `${formatNumber(value)}°`;
  }
  if ((kind ?? "") === "measurement" || (metricKey ?? "").startsWith("raw_") || (metricKey ?? "").includes("_width") || (metricKey ?? "").includes("_height") || (metricKey ?? "").includes("_distance") || (metricKey ?? "").includes("_offset")) {
    return `${formatNumber(value)} px`;
  }
  return formatNumber(value);
}

function segmentTooltip(segment: VisualizationLineResponse): string {
  const pieces = [segment.label ?? formatLabel(segment.key)];
  if (segment.metric_key) {
    const value = metricLabel(segment.metric_key, segment.kind);
    if (value) {
      pieces.push(value);
    }
  }
  return pieces.join(" · ");
}

function segmentClassName(segment: VisualizationLineResponse): string {
  const kind = segment.kind ?? "measurement";
  if (kind === "angle" || kind === "axis") {
    return "overlay-line--angle";
  }
  if (kind === "guide") {
    return "overlay-line--thirds";
  }
  if (kind === "symmetry") {
    return "overlay-line--symmetry";
  }
  return "overlay-line--measurement";
}

function labelClassName(segment: VisualizationLineResponse): string {
  const kind = segment.kind ?? "measurement";
  if (kind === "angle" || kind === "axis") {
    return "overlay-label--angle";
  }
  if (kind === "guide") {
    return "overlay-label--thirds";
  }
  if (kind === "symmetry") {
    return "overlay-label--symmetry";
  }
  return "overlay-label--measurement";
}

function buildSegments(source: VisualizationLineResponse[] | undefined): OverlaySegment[] {
  if (!source) {
    return [];
  }
  return source
    .map((segment) => {
      const start = allPoints.value[segment.start];
      const end = allPoints.value[segment.end];
      if (!start || !end) {
        return null;
      }
      if (!segmentLayerVisible(segment, layers)) {
        return null;
      }
      const labelOffset = segment.kind === "guide"
        ? overlaySizes.value.guideLabelOffset
        : (segment.kind === "angle" || segment.kind === "axis"
            ? -overlaySizes.value.angleLabelOffset
            : -overlaySizes.value.lineLabelOffset);
      return {
        id: segment.key,
        start,
        end,
        label: metricLabel(segment.metric_key, segment.kind),
        tooltip: segmentTooltip(segment),
        labelPoint: midpoint(start, end, labelOffset),
        className: segmentClassName(segment),
        labelClassName: labelClassName(segment),
        supportedFeatures: segment.supported_features ?? [],
      };
    })
    .filter((segment): segment is OverlaySegment => Boolean(segment));
}

const visibleSegments = computed<OverlaySegment[]>(() => [
  ...buildSegments(getMeasurementOverlaySegments(visualization.value, "guides")),
  ...buildSegments(getMeasurementOverlaySegments(visualization.value, "axes")),
  ...buildSegments(getMeasurementOverlaySegments(visualization.value, "lines")),
]);

const featureDescriptions = computed<Record<string, string>>(() => (
  props.locale === "ru"
    ? {
        ratio_eye_distance_to_mouth_width: "Сопоставляет межглазное расстояние и ширину рта.",
        ratio_intercanthal_to_eye_width: "Сопоставляет межкантальное расстояние и среднюю ширину глаза.",
        ratio_outercanthal_to_face_width: "Показывает отношение внешнего расстояния между глазами к ширине лица.",
        ratio_nose_width_to_face_width: "Сопоставляет ширину носа и ширину лица.",
        ratio_mouth_width_to_nose_width: "Сопоставляет ширину рта и ширину носа.",
        sym_eye_nose_delta_ratio: "Показывает разбалансировку левого и правого расстояния от глаза до носа.",
        sym_mouth_nose_delta_ratio: "Показывает разбалансировку левого и правого расстояния от уголка рта до носа.",
        nose_tip_midline_offset: "Смещение кончика носа относительно центральной оси лица.",
        mouth_center_midline_offset: "Смещение центра рта относительно центральной оси лица.",
        chin_center_midline_offset: "Смещение центра подбородка относительно центральной оси лица.",
        brow_to_eye_center_height: "Высота от линии бровей до центра глаз.",
        eye_center_to_nose_base_height: "Высота от центра глаз до основания носа.",
        nose_base_to_mouth_center_height: "Высота от основания носа до центра рта.",
        mouth_center_to_chin_height: "Высота от центра рта до подбородка.",
        canthal_tilt_deg: "Наклон глазной линии.",
        angle_mouth_line_deg: "Наклон линии рта.",
        nose_axis_angle_deg: "Направление оси носа.",
        philtrum_axis_angle_deg: "Направление оси филтрума.",
        brow_line_angle_deg: "Наклон линии бровей.",
        jaw_angle_asymmetry: "Разница между левым и правым углом нижней челюсти.",
      }
    : {
        ratio_eye_distance_to_mouth_width: "Compares interpupillary distance against mouth width.",
        ratio_intercanthal_to_eye_width: "Compares intercanthal spacing against mean eye width.",
        ratio_outercanthal_to_face_width: "Shows the outer eye span relative to face width.",
        ratio_nose_width_to_face_width: "Compares nose width against face width.",
        ratio_mouth_width_to_nose_width: "Compares mouth width against nose width.",
        sym_eye_nose_delta_ratio: "Shows the imbalance between left and right eye-to-nose distances.",
        sym_mouth_nose_delta_ratio: "Shows the imbalance between left and right mouth-corner-to-nose distances.",
        nose_tip_midline_offset: "Shows how far the nose tip sits from the facial center axis.",
        mouth_center_midline_offset: "Shows how far the mouth center sits from the facial center axis.",
        chin_center_midline_offset: "Shows how far the chin center sits from the facial center axis.",
        brow_to_eye_center_height: "Vertical height from brow line to eye center.",
        eye_center_to_nose_base_height: "Vertical height from eye center to nose base.",
        nose_base_to_mouth_center_height: "Vertical height from nose base to mouth center.",
        mouth_center_to_chin_height: "Vertical height from mouth center to chin.",
        canthal_tilt_deg: "Tilt of the eye line.",
        angle_mouth_line_deg: "Tilt of the mouth line.",
        nose_axis_angle_deg: "Direction of the nose axis.",
        philtrum_axis_angle_deg: "Direction of the philtrum axis.",
        brow_line_angle_deg: "Tilt of the brow line.",
        jaw_angle_asymmetry: "Difference between left and right lower-jaw angular proxies.",
      }
));

const featurePrimitiveMap: Record<string, string[]> = {
  ratio_eye_distance_to_mouth_width: ["interpupillary_line", "mouth_width_line"],
  ratio_intercanthal_to_eye_width: ["intercanthal_line", "left_eye_width_line", "right_eye_width_line"],
  ratio_outercanthal_to_face_width: ["outercanthal_line"],
  ratio_nose_width_to_face_width: ["nose_width_line"],
  ratio_mouth_width_to_nose_width: ["mouth_width_line", "nose_width_line"],
  sym_eye_nose_delta_ratio: ["left_eye_to_nose", "right_eye_to_nose"],
  sym_mouth_nose_delta_ratio: ["left_mouth_to_nose", "right_mouth_to_nose"],
  nose_tip_midline_offset: ["nose_midline_projection_line"],
  mouth_center_midline_offset: ["mouth_midline_projection_line"],
  chin_center_midline_offset: ["chin_midline_projection_line"],
  brow_to_eye_center_height: ["brow_to_eye_center_line"],
  eye_center_to_nose_base_height: ["eye_center_to_nose_base_line"],
  nose_base_to_mouth_center_height: ["nose_base_to_mouth_center_line"],
  mouth_center_to_chin_height: ["mouth_center_to_chin_line"],
  canthal_tilt_deg: ["eye_line_angle"],
  angle_mouth_line_deg: ["mouth_line_angle"],
  nose_axis_angle_deg: ["nose_axis"],
  philtrum_axis_angle_deg: ["philtrum_axis"],
  brow_line_angle_deg: ["brow_line"],
  jaw_angle_asymmetry: ["jaw_left_axis", "jaw_right_axis"],
};

function describeFeatureDeviation(feature: FeatureDeviation): HighlightItem | null {
  const primitiveIds = featurePrimitiveMap[feature.feature_name] ?? [];
  if (!primitiveIds.length) {
    return null;
  }
  return {
    featureName: feature.feature_name,
    value: `z=${formatNumber(feature.z_score)}`,
    description: featureDescriptions.value[feature.feature_name] ?? feature.direction,
    primitiveIds,
  };
}

const highlightItems = computed<HighlightItem[]>(() => {
  const features = props.report?.feature_level_deviations.top_contributing_features ?? [];
  return features
    .map((feature) => describeFeatureDeviation(feature))
    .filter((item): item is HighlightItem => Boolean(item));
});
</script>
