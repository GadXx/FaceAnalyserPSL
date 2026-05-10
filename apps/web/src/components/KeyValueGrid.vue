<template>
  <div class="kv-grid">
    <div v-for="[key, value] in rows" :key="key" class="kv-grid__item">
      <p class="kv-grid__label">
        <MetricLabel :label="formatLabel(key)" :help-key="key" :locale="locale" />
      </p>
      <p class="kv-grid__value">{{ formatValue(value) }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

import { formatLabel, formatValue } from "../utils/format";
import type { Locale } from "../utils/i18n";
import MetricLabel from "./MetricLabel.vue";

const props = withDefaults(defineProps<{
  value: Record<string, unknown> | null | undefined;
  locale?: Locale;
}>(), {
  locale: "en",
});

const rows = computed(() => Object.entries(props.value ?? {}));
const locale = computed(() => props.locale);
</script>
