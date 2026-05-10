<template>
  <details class="json-debug">
    <summary>Raw JSON debug</summary>
    <pre>{{ formatted }}</pre>
  </details>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  value: unknown;
}>();

const formatted = computed(() => JSON.stringify(props.value, null, 2));
</script>
