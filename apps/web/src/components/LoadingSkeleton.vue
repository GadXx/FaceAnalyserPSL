<template>
  <div class="skeleton-grid">
    <div v-for="index in 4" :key="index" class="skeleton-card">
      <div class="skeleton-card__line skeleton-card__line--title"></div>
      <div class="skeleton-card__line"></div>
      <div class="skeleton-card__line"></div>
      <div class="skeleton-card__line skeleton-card__line--short"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
</script>
