<template>
  <div class="deviation-list">
    <article v-for="item in items" :key="item.feature_name" class="deviation-list__item">
      <div>
        <p class="deviation-list__label">
          <MetricLabel :label="item.feature_name" :help-key="item.feature_name" :locale="locale" />
        </p>
        <p class="deviation-list__direction">{{ item.direction }}</p>
      </div>
      <div class="deviation-list__metrics">
        <span class="deviation-list__metric">z={{ formatNumber(item.z_score) }}</span>
        <span class="deviation-list__metric deviation-list__metric--strong">|z|={{ formatNumber(item.abs_z_score) }}</span>
      </div>
    </article>
  </div>
</template>

<script setup lang="ts">
import type { FeatureDeviation } from "../types/api";
import { formatNumber } from "../utils/format";
import type { Locale } from "../utils/i18n";
import MetricLabel from "./MetricLabel.vue";

withDefaults(defineProps<{
  items: FeatureDeviation[];
  locale?: Locale;
}>(), {
  locale: "en",
});
</script>
