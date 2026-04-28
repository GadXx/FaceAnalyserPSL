<template>
  <section class="section-card">
    <header class="section-card__header">
      <div>
        <p v-if="eyebrow" class="section-card__eyebrow">{{ eyebrow }}</p>
        <h2 class="section-card__title">
          <MetricLabel :label="title" :help-key="helpKey" :locale="locale" />
        </h2>
      </div>
      <slot name="header-extra" />
    </header>
    <p v-if="description" class="section-card__description">{{ description }}</p>
    <div class="section-card__body">
      <slot />
    </div>
  </section>
</template>

<script setup lang="ts">
import type { Locale } from "../utils/i18n";
import MetricLabel from "./MetricLabel.vue";

withDefaults(defineProps<{
  title: string;
  description?: string;
  eyebrow?: string;
  helpKey?: string;
  locale?: Locale;
}>(), {
  locale: "en",
});
</script>
