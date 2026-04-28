export type Locale = "ru" | "en";

export interface Messages {
  languageLabel: string;
  languageRu: string;
  languageEn: string;
  appEyebrow: string;
  appTitle: string;
  appDescription: string;
  backendStatusTitle: string;
  backendStatusEyebrow: string;
  backendChecking: string;
  healthAvailable: string;
  healthMissing: string;
  healthChecking: string;
  healthUnknown: string;
  uploadTitle: string;
  uploadEyebrow: string;
  uploadDropTitle: string;
  uploadDropDescription: string;
  uploadChoose: string;
  uploadClear: string;
  unknownContentType: string;
  analysisTitle: string;
  analysisEyebrow: string;
  analysisDescription: string;
  referenceGroup: string;
  referenceGroupHelp: string;
  referenceGroupPlaceholder: string;
  manualGroup: string;
  manualGroupHelp: string;
  manualGroupPlaceholder: string;
  executionProvider: string;
  executionProviderHelp: string;
  deviceId: string;
  deviceIdHelp: string;
  outputDir: string;
  outputDirHelp: string;
  strictGpu: string;
  strictGpuHelp: string;
  analyzing: string;
  runAnalysis: string;
  requestError: string;
  contextTitle: string;
  contextEyebrow: string;
  contextDescription: string;
  qualityTitle: string;
  qualityEyebrow: string;
  qualityDescription: string;
  visualAnalysisTitle: string;
  visualAnalysisEyebrow: string;
  visualAnalysisDescription: string;
  rawMeasurementsTitle: string;
  normalizedRatiosTitle: string;
  symmetryMetricsTitle: string;
  angleMetricsTitle: string;
  geometryEyebrow: string;
  featuresEyebrow: string;
  referenceTitle: string;
  referenceEyebrow: string;
  deviationsTitle: string;
  deviationsEyebrow: string;
  deviationsDescription: string;
  compositeTitle: string;
  compositeEyebrow: string;
  limitationsTitle: string;
  limitationsEyebrow: string;
  artifactsTitle: string;
  artifactsEyebrow: string;
  emptyTitle: string;
  emptyEyebrow: string;
  emptyDescription: string;
  emptyBody: string;
  previewAlt: string;
  statusKey: string;
  canAnalyzeKey: string;
  dependencyCountKey: string;
  selectedFileName: string;
  selectedFileSizeMb: string;
  selectedFileType: string;
  inputSource: string;
  uploadOriginalFilename: string;
  uploadContentType: string;
  outputDirectoryUsed: string;
  referenceGroupUsed: string;
  executionProviderUsed: string;
  deviceIdUsed: string;
  resultImageSource: string;
  analysisImagePath: string;
  normalizedFrameImagePath: string;
  annotatedImagePath: string;
  alignedImagePath: string;
  coordinateSpace: string;
}

export const messages: Record<Locale, Messages> = {
  en: {
    languageLabel: "Language",
    languageRu: "Russian",
    languageEn: "English",
    appEyebrow: "Objective morphometric system",
    appTitle: "Face geometry analysis workspace",
    appDescription:
      "Upload an image, validate preprocessing quality, compare normalized measurements against reference groups, and inspect feature-level deviations without any subjective scoring.",
    backendStatusTitle: "Backend status",
    backendStatusEyebrow: "Connection",
    backendChecking: "Checking backend health…",
    healthAvailable: "available",
    healthMissing: "missing",
    healthChecking: "checking",
    healthUnknown: "unknown",
    uploadTitle: "Upload image",
    uploadEyebrow: "Input",
    uploadDropTitle: "Drop a face image here",
    uploadDropDescription: "JPEG, PNG, WEBP, or BMP up to 15 MB. The image is sent directly to the morphometric API.",
    uploadChoose: "Choose file",
    uploadClear: "Clear",
    unknownContentType: "Unknown content type",
    analysisTitle: "Analysis request",
    analysisEyebrow: "Controls",
    analysisDescription: "These fields control which reference group is used and how the backend executes preprocessing.",
    referenceGroup: "Reference group",
    referenceGroupHelp: "Select a known reference bucket from the trained reference model. This affects z-scores, percentiles, and Mahalanobis distance.",
    referenceGroupPlaceholder: "Use sidecar metadata if available",
    manualGroup: "Manual group override",
    manualGroupHelp: "Optional explicit override in backend format. If filled, it takes priority over the dropdown and sidecar metadata.",
    manualGroupPlaceholder: "sex_group_normalized=Female,age_group_normalized=20-29,…",
    executionProvider: "Execution provider",
    executionProviderHelp: "Choose where preprocessing runs. CPU is the safest default. CUDA uses GPU if the backend runtime is ready.",
    deviceId: "Device id",
    deviceIdHelp: "GPU device index for CUDA mode. Keep 0 unless the backend host has multiple visible GPUs.",
    outputDir: "Output directory override",
    outputDirHelp: "Optional backend path for generated reports. Leave empty to use the backend default output location.",
    strictGpu: "Strict GPU mode",
    strictGpuHelp: "If enabled, the request fails instead of silently falling back to CPU when CUDA is unavailable.",
    analyzing: "Analyzing…",
    runAnalysis: "Run morphometric analysis",
    requestError: "Request error.",
    contextTitle: "Input and request context",
    contextEyebrow: "Context",
    contextDescription: "Only the most relevant request and file metadata are shown here instead of the full raw payload.",
    qualityTitle: "Quality and preprocessing",
    qualityEyebrow: "Quality",
    qualityDescription: "Detection confidence, pose, quality gate result, and preprocessing status.",
    visualAnalysisTitle: "Morphometric overlay",
    visualAnalysisEyebrow: "Visual analysis",
    visualAnalysisDescription: "Overlay the real face box, anchor landmarks, measurement lines, symmetry axis, and angle cues on top of the uploaded image.",
    rawMeasurementsTitle: "Raw measurements",
    normalizedRatiosTitle: "Normalized ratios",
    symmetryMetricsTitle: "Symmetry metrics",
    angleMetricsTitle: "Angle metrics",
    geometryEyebrow: "Geometry",
    featuresEyebrow: "Features",
    referenceTitle: "Reference comparison",
    referenceEyebrow: "Reference model",
    deviationsTitle: "Feature deviations",
    deviationsEyebrow: "Top contributors",
    deviationsDescription: "Largest per-feature standardized deviations from the selected reference group.",
    compositeTitle: "Composite indices",
    compositeEyebrow: "Optional indices",
    limitationsTitle: "Limitations",
    limitationsEyebrow: "Interpretation",
    artifactsTitle: "Artifacts and generated files",
    artifactsEyebrow: "Outputs",
    emptyTitle: "Ready for first analysis",
    emptyEyebrow: "Empty state",
    emptyDescription: "Upload a face image, choose a reference group, and run the existing backend pipeline through the file-upload API.",
    emptyBody: "The interface is connected to the real morphometric API and shows structured results instead of raw JSON output.",
    previewAlt: "Selected file preview",
    statusKey: "status",
    canAnalyzeKey: "can_analyze",
    dependencyCountKey: "dependency_count",
    selectedFileName: "selected_file_name",
    selectedFileSizeMb: "selected_file_size_mb",
    selectedFileType: "selected_file_type",
    inputSource: "input_source",
    uploadOriginalFilename: "upload_original_filename",
    uploadContentType: "upload_content_type",
    outputDirectoryUsed: "output_directory_used",
    referenceGroupUsed: "reference_group_used",
    executionProviderUsed: "execution_provider_used",
    deviceIdUsed: "device_id_used",
    resultImageSource: "result_image_source",
    analysisImagePath: "analysis_image_path",
    normalizedFrameImagePath: "normalized_frame_image_path",
    annotatedImagePath: "annotated_image_path",
    alignedImagePath: "aligned_image_path",
    coordinateSpace: "coordinate_space",
  },
  ru: {
    languageLabel: "Язык",
    languageRu: "Русский",
    languageEn: "Английский",
    appEyebrow: "Система объективной морфометрии",
    appTitle: "Рабочее пространство анализа геометрии лица",
    appDescription:
      "Загрузите изображение, проверьте качество preprocessing, сравните нормализованные измерения с reference-группой и изучите отклонения признаков без каких-либо субъективных оценок.",
    backendStatusTitle: "Состояние backend",
    backendStatusEyebrow: "Подключение",
    backendChecking: "Проверка доступности backend…",
    healthAvailable: "доступно",
    healthMissing: "отсутствует",
    healthChecking: "проверка",
    healthUnknown: "неизвестно",
    uploadTitle: "Загрузка изображения",
    uploadEyebrow: "Вход",
    uploadDropTitle: "Перетащите изображение лица сюда",
    uploadDropDescription: "JPEG, PNG, WEBP или BMP до 15 МБ. Файл отправляется напрямую в morphometric API.",
    uploadChoose: "Выбрать файл",
    uploadClear: "Очистить",
    unknownContentType: "Неизвестный тип файла",
    analysisTitle: "Параметры анализа",
    analysisEyebrow: "Управление",
    analysisDescription: "Эти поля управляют выбором reference-группы и тем, как backend запускает preprocessing.",
    referenceGroup: "Reference-группа",
    referenceGroupHelp: "Выберите готовую reference-группу из обученной reference-модели. Это влияет на z-score, percentiles и Mahalanobis distance.",
    referenceGroupPlaceholder: "Использовать sidecar metadata, если она есть",
    manualGroup: "Ручной override группы",
    manualGroupHelp: "Необязательный явный override в backend-формате. Если поле заполнено, оно имеет приоритет над выпадающим списком и sidecar metadata.",
    manualGroupPlaceholder: "sex_group_normalized=Female,age_group_normalized=20-29,…",
    executionProvider: "Execution provider",
    executionProviderHelp: "Выбор среды для preprocessing. CPU — самый безопасный вариант. CUDA использует GPU, если runtime на backend готов.",
    deviceId: "Device id",
    deviceIdHelp: "Индекс GPU для CUDA-режима. Обычно оставляют 0, если на backend-машине видна одна видеокарта.",
    outputDir: "Override директории результатов",
    outputDirHelp: "Необязательный путь на backend для сохранения отчётов. Если оставить пустым, используется путь по умолчанию.",
    strictGpu: "Строгий GPU-режим",
    strictGpuHelp: "Если включено, запрос завершится ошибкой вместо тихого fallback на CPU, когда CUDA недоступна.",
    analyzing: "Анализ…",
    runAnalysis: "Запустить морфометрический анализ",
    requestError: "Ошибка запроса.",
    contextTitle: "Контекст входа и запроса",
    contextEyebrow: "Контекст",
    contextDescription: "Здесь показываются только важные метаданные запроса и файла, без полного сырого payload.",
    qualityTitle: "Качество и preprocessing",
    qualityEyebrow: "Качество",
    qualityDescription: "Уверенность детекции, pose, результат quality gate и статус preprocessing.",
    visualAnalysisTitle: "Морфометрический overlay",
    visualAnalysisEyebrow: "Визуальный анализ",
    visualAnalysisDescription: "Наложение реального face box, опорных landmarks, линий измерений, оси симметрии и угловых ориентиров поверх загруженного изображения.",
    rawMeasurementsTitle: "Сырые измерения",
    normalizedRatiosTitle: "Нормализованные отношения",
    symmetryMetricsTitle: "Метрики симметрии",
    angleMetricsTitle: "Угловые метрики",
    geometryEyebrow: "Геометрия",
    featuresEyebrow: "Признаки",
    referenceTitle: "Сравнение с reference",
    referenceEyebrow: "Reference-модель",
    deviationsTitle: "Отклонения признаков",
    deviationsEyebrow: "Главные contributors",
    deviationsDescription: "Наибольшие стандартизованные отклонения признаков относительно выбранной reference-группы.",
    compositeTitle: "Композитные индексы",
    compositeEyebrow: "Опциональные индексы",
    limitationsTitle: "Ограничения",
    limitationsEyebrow: "Интерпретация",
    artifactsTitle: "Артефакты и созданные файлы",
    artifactsEyebrow: "Выходы",
    emptyTitle: "Готово к первому анализу",
    emptyEyebrow: "Пустое состояние",
    emptyDescription: "Загрузите изображение лица, выберите reference-группу и запустите существующий backend pipeline через file-upload API.",
    emptyBody: "Интерфейс подключён к реальному morphometric API и показывает структурированный результат вместо простого raw JSON.",
    previewAlt: "Предпросмотр выбранного файла",
    statusKey: "status",
    canAnalyzeKey: "can_analyze",
    dependencyCountKey: "dependency_count",
    selectedFileName: "имя_выбранного_файла",
    selectedFileSizeMb: "размер_файла_мб",
    selectedFileType: "тип_файла",
    inputSource: "источник_входа",
    uploadOriginalFilename: "исходное_имя_загрузки",
    uploadContentType: "тип_контента_загрузки",
    outputDirectoryUsed: "использованная_директория_вывода",
    referenceGroupUsed: "использованная_reference_группа",
    executionProviderUsed: "использованный_execution_provider",
    deviceIdUsed: "использованный_device_id",
    resultImageSource: "источник_результирующего_изображения",
    analysisImagePath: "путь_к_analysis_image",
    normalizedFrameImagePath: "путь_к_normalized_frame_image",
    annotatedImagePath: "путь_к_annotated_image",
    alignedImagePath: "путь_к_aligned_image",
    coordinateSpace: "система_координат_overlay",
  },
};
