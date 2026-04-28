<template>
  <span
    v-if="entry"
    ref="rootRef"
    class="info-tooltip"
    @mouseenter="openTooltip"
    @mouseleave="closeTooltip"
  >
    <button
      type="button"
      class="info-tooltip__trigger"
      :aria-label="ariaLabel"
      :aria-expanded="isOpen ? 'true' : 'false'"
      :aria-describedby="tooltipId"
      @focus="openTooltip"
      @blur="closeTooltip"
      @click.stop="toggleTooltip"
      @keydown.escape.stop.prevent="closeTooltip"
    >
      i
    </button>
    <span v-if="isOpen" :id="tooltipId" class="info-tooltip__bubble" role="tooltip">
      <strong v-if="entry.label" class="info-tooltip__heading">{{ entry.label }}</strong>
      <span class="info-tooltip__line">{{ entry.short }}</span>
      <span v-if="entry.extended" class="info-tooltip__line">{{ entry.extended }}</span>
      <span v-if="entry.interpretation" class="info-tooltip__line info-tooltip__line--hint">
        {{ entry.interpretation }}
      </span>
      <span v-if="entry.warning" class="info-tooltip__line info-tooltip__line--warning">
        {{ entry.warning }}
      </span>
    </span>
  </span>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";

import { getHelpEntry } from "../content/helpText";
import type { Locale } from "../utils/i18n";

const props = defineProps<{
  helpKey?: string;
  locale: Locale;
}>();

const entry = computed(() => getHelpEntry(props.helpKey, props.locale));
const isOpen = ref(false);
const tooltipId = `info-tooltip-${Math.random().toString(36).slice(2, 10)}`;

const ariaLabel = computed(() =>
  props.locale === "ru"
    ? "Показать пояснение к метрике или секции"
    : "Show explanation for this metric or section",
);

function openTooltip(): void {
  if (!entry.value) {
    return;
  }
  isOpen.value = true;
}

function closeTooltip(): void {
  isOpen.value = false;
}

function toggleTooltip(): void {
  if (!entry.value) {
    return;
  }
  isOpen.value = !isOpen.value;
}
</script>
