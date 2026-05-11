<template>
  <div class="expanded-geometry">
    <div v-if="!expandedGeometry?.available" class="alert alert--info">
      <strong>{{ copy.unavailableTitle }}</strong>
      <span>{{ copy.unavailableBody }}</span>
    </div>

    <template v-else>
      <KeyValueGrid
        :locale="locale"
        :value="{
          [copy.totalLabel]: expandedGeometry.mediapipe_feature_count,
          [copy.nonNullLabel]: expandedGeometry.mediapipe_non_null_feature_count,
        }"
      />

      <div v-if="expandedGeometry.reliability_counts" class="expanded-geometry__reliability">
        <span
          v-for="(count, key) in expandedGeometry.reliability_counts"
          :key="key"
          class="expanded-geometry__badge"
          :class="`expanded-geometry__badge--${key}`"
        >
          {{ key }}: {{ count }}
        </span>
      </div>

      <div class="expanded-geometry__groups">
        <article
          v-for="[groupName, rows] in groupedRows"
          :key="groupName"
          class="expanded-geometry__group"
        >
          <h4>{{ resolveGroupTitle(groupName) }}</h4>
          <ul class="expanded-geometry__rows">
            <li v-for="row in rows" :key="row.feature_name" class="expanded-geometry__row">
              <div class="expanded-geometry__row-head">
                <MetricLabel :label="row.feature_name" :help-key="row.feature_name" :locale="locale" />
                <span class="expanded-geometry__value">{{ formatFeatureValue(row.value) }}</span>
              </div>
              <div class="expanded-geometry__row-meta">
                <span class="expanded-geometry__meta-pill">{{ row.source ?? "unknown" }}</span>
                <span class="expanded-geometry__meta-pill" :class="`expanded-geometry__meta-pill--${row.reliability ?? 'unknown'}`">
                  {{ row.reliability ?? "unknown" }}
                </span>
              </div>
            </li>
          </ul>
        </article>
      </div>

      <div v-if="unavailableItems.length" class="expanded-geometry__unavailable">
        <strong>{{ copy.faceParsingNoteTitle }}</strong>
        <ul>
          <li v-for="item in unavailableItems" :key="item">{{ item }}</li>
        </ul>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import type { ExpandedGeometryFeatureRow, ExpandedGeometryResponse } from "../types/api";
import { formatNumber } from "../utils/format";
import type { Locale } from "../utils/i18n";
import KeyValueGrid from "./KeyValueGrid.vue";
import MetricLabel from "./MetricLabel.vue";

const props = defineProps<{
  expandedGeometry?: ExpandedGeometryResponse | null;
  locale: Locale;
}>();

const copy = computed(() => (props.locale === "ru"
  ? {
      unavailableTitle: "Расширенная геометрия недоступна.",
      unavailableBody: "MediaPipe-признаки не были получены для текущего результата.",
      totalLabel: "expanded_mediapipe_feature_count",
      nonNullLabel: "expanded_mediapipe_non_null_count",
      faceParsingNoteTitle: "Недоступно без face parsing:",
      groups: {
        face_oval: "Овал лица",
        eyes: "Глаза",
        brows: "Брови",
        nose: "Нос",
        mouth_lips: "Рот/губы",
        cheekbones_midface: "Скулы/средняя зона",
        jaw_chin: "Челюсть/подбородок",
      } as Record<string, string>,
    }
  : {
      unavailableTitle: "Expanded geometry is unavailable.",
      unavailableBody: "MediaPipe-derived features are not available for this result.",
      totalLabel: "expanded_mediapipe_feature_count",
      nonNullLabel: "expanded_mediapipe_non_null_count",
      faceParsingNoteTitle: "Unavailable without face parsing:",
      groups: {
        face_oval: "Face oval",
        eyes: "Eyes",
        brows: "Brows",
        nose: "Nose",
        mouth_lips: "Mouth/lips",
        cheekbones_midface: "Cheekbones/midface",
        jaw_chin: "Jaw/chin",
      } as Record<string, string>,
    }));

const groupedRows = computed<Array<[string, ExpandedGeometryFeatureRow[]]>>(() => {
  const groups = props.expandedGeometry?.groups ?? {};
  return Object.entries(groups);
});

const unavailableItems = computed<string[]>(() =>
  (props.expandedGeometry?.unavailable_without_face_parsing ?? [])
    .map((item) => String(item?.feature_name ?? ""))
    .filter((item) => item.length > 0),
);

function resolveGroupTitle(groupName: string): string {
  return copy.value.groups[groupName] ?? groupName;
}

function formatFeatureValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return formatNumber(value, 4);
}
</script>

<style scoped>
.expanded-geometry {
  display: grid;
  gap: 0.9rem;
}

.expanded-geometry__reliability,
.expanded-geometry__row-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.expanded-geometry__badge,
.expanded-geometry__meta-pill {
  border-radius: 999px;
  padding: 0.2rem 0.55rem;
  font-size: 0.72rem;
  background: rgba(16, 46, 79, 0.08);
}

.expanded-geometry__meta-pill--reliable,
.expanded-geometry__badge--reliable {
  background: rgba(33, 92, 68, 0.14);
  color: #215c44;
}

.expanded-geometry__meta-pill--proxy,
.expanded-geometry__badge--proxy {
  background: rgba(181, 107, 45, 0.15);
  color: #8f5120;
}

.expanded-geometry__meta-pill--unstable,
.expanded-geometry__badge--unstable {
  background: rgba(168, 62, 40, 0.14);
  color: #8d301d;
}

.expanded-geometry__groups {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.75rem;
}

.expanded-geometry__group {
  border: 1px solid var(--color-line);
  border-radius: 1rem;
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.5);
}

.expanded-geometry__group h4 {
  margin: 0 0 0.6rem;
}

.expanded-geometry__rows {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 0.55rem;
}

.expanded-geometry__row {
  border-top: 1px dashed var(--color-line);
  padding-top: 0.45rem;
}

.expanded-geometry__row:first-child {
  border-top: 0;
  padding-top: 0;
}

.expanded-geometry__row-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.6rem;
}

.expanded-geometry__value {
  font-weight: 600;
}

.expanded-geometry__unavailable strong {
  display: block;
  margin-bottom: 0.3rem;
}

.expanded-geometry__unavailable ul {
  margin: 0;
  padding-left: 1rem;
}
</style>
