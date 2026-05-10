<template>
  <span class="metric-label">
    <span>{{ resolvedLabel }}</span>
    <InfoTooltip :help-key="helpKey" :locale="locale" />
  </span>
</template>

<script setup lang="ts">
import { computed } from "vue";

import { getHelpLabel } from "../content/helpText";
import type { Locale } from "../utils/i18n";
import InfoTooltip from "./InfoTooltip.vue";

const props = defineProps<{
  label: string;
  helpKey?: string;
  locale: Locale;
}>();

const resolvedLabel = computed(() => getHelpLabel(props.helpKey ?? props.label, props.locale, props.label));
</script>
