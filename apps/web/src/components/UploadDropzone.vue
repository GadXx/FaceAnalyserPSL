<template>
  <div
    class="upload-dropzone"
    :class="{ 'upload-dropzone--active': isDragActive, 'upload-dropzone--error': errorMessage }"
    @dragenter.prevent="isDragActive = true"
    @dragover.prevent="isDragActive = true"
    @dragleave.prevent="isDragActive = false"
    @drop.prevent="handleDrop"
  >
    <input
      ref="inputRef"
      class="upload-dropzone__input"
      type="file"
      accept="image/jpeg,image/png,image/webp,image/bmp"
      @change="handleFileChange"
    />
    <div class="upload-dropzone__content">
      <div class="upload-dropzone__icon">+</div>
      <h3>{{ title }}</h3>
      <p>{{ description }}</p>
      <div class="upload-dropzone__actions">
        <button class="button button--primary" type="button" @click="openFilePicker">{{ chooseLabel }}</button>
        <button class="button button--ghost" type="button" @click="$emit('clear')" :disabled="!hasFile">{{ clearLabel }}</button>
      </div>
      <p v-if="errorMessage" class="upload-dropzone__error">{{ errorMessage }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

const emit = defineEmits<{
  select: [file: File];
  clear: [];
}>();

defineProps<{
  errorMessage?: string | null;
  hasFile: boolean;
  title: string;
  description: string;
  chooseLabel: string;
  clearLabel: string;
}>();

const inputRef = ref<HTMLInputElement | null>(null);
const isDragActive = ref(false);

function openFilePicker(): void {
  inputRef.value?.click();
}

function handleFileChange(event: Event): void {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (file) {
    emit("select", file);
  }
  target.value = "";
}

function handleDrop(event: DragEvent): void {
  isDragActive.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    emit("select", file);
  }
}
</script>
