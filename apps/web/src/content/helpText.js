/**
 * @typedef {"ru" | "en"} Locale
 */

/**
 * @typedef {{
 *   label?: string;
 *   short: string;
 *   extended?: string;
 *   interpretation?: string;
 *   warning?: string;
 * }} LocalizedHelpEntry
 */

/**
 * @typedef {{
 *   en: LocalizedHelpEntry;
 *   ru: LocalizedHelpEntry;
 * }} HelpEntry
 */

/** @type {Record<string, string>} */
const HELP_ALIASES = {
  "section.context": "section.context",
  "section.analysis_request": "section.analysis_request",
  "section.quality": "section.quality",
  "section.visual_analysis": "section.visual_analysis",
  "section.raw_measurements": "section.raw_measurements",
  "section.normalized_ratios": "section.normalized_ratios",
  "section.symmetry_metrics": "section.symmetry_metrics",
  "section.angle_metrics": "section.angle_metrics",
  "section.reference_comparison": "section.reference_comparison",
  "section.feature_deviations": "section.feature_deviations",
  "section.composite_indices": "section.composite_indices",
  "section.limitations": "section.limitations",
  "section.artifacts": "section.artifacts",
  "field.reference_group_selector": "field.reference_group_selector",
  "field.manual_group_override": "field.manual_group_override",
  "field.execution_provider": "field.execution_provider",
  "field.device_id": "field.device_id",
  "field.output_dir": "field.output_dir",
  "field.strict_gpu": "field.strict_gpu",
  "toggle.bbox": "toggle.bbox",
  "toggle.landmarks": "toggle.landmarks",
  "toggle.raw_measurements": "toggle.raw_measurements",
  "toggle.eyes": "toggle.eyes",
  "toggle.nose": "toggle.nose",
  "toggle.mouth": "toggle.mouth",
  "toggle.chin_jaw": "toggle.chin_jaw",
  "toggle.thirds": "toggle.thirds",
  "toggle.debug_mode": "toggle.debug_mode",
  "toggle.symmetry": "toggle.symmetry",
  "toggle.angles": "toggle.angles",
  "toggle.labels": "toggle.labels",
  "toggle.all_landmarks": "toggle.all_landmarks",
  "toggle.debug_indices": "toggle.debug_indices",
  processing_status: "processing_status",
  quality_pass: "quality_pass",
  quality_fail_reason: "quality_fail_reason",
  processing_error_message: "processing_error_message",
  det_score: "det_score",
  yaw: "pose_yaw_deg",
  pitch: "pose_pitch_deg",
  roll: "pose_roll_deg",
  num_faces: "num_faces",
  blur_variance: "blur_variance",
  occlusion_score: "occlusion_score",
  raw_face_width_px: "raw_face_width_px",
  raw_face_height_px: "raw_face_height_px",
  raw_eye_distance_px: "raw_eye_distance_px",
  raw_mouth_width_px: "raw_mouth_width_px",
  raw_eye_to_nose_px: "raw_eye_to_nose_px",
  raw_nose_to_mouth_px: "raw_nose_to_mouth_px",
  raw_eye_to_mouth_px: "raw_eye_to_mouth_px",
  raw_nose_center_offset_px: "raw_nose_center_offset_px",
  raw_mouth_center_offset_px: "raw_mouth_center_offset_px",
  ratio_face_aspect: "ratio_face_aspect",
  ratio_eye_distance_to_mouth_width: "ratio_eye_distance_to_mouth_width",
  symmetry_index: "symmetry_index",
  sym_eye_nose_delta_ratio: "sym_eye_nose_delta_ratio",
  sym_mouth_nose_delta_ratio: "sym_mouth_nose_delta_ratio",
  angle_eye_line_deg: "angle_eye_line_deg",
  angle_mouth_line_deg: "angle_mouth_line_deg",
  angle_eye_center_to_mouth_center_deg: "angle_eye_center_to_mouth_center_deg",
  angle_nose_to_mouth_center_deg: "angle_nose_to_mouth_center_deg",
  group_key: "reference_group",
  sample_count: "sample_count",
  complete_case_count: "complete_case_count",
  mahalanobis_distance: "mahalanobis_distance",
  percentile_positions: "percentile_position",
  mean_absolute_z_score: "mean_absolute_z_score",
  proportion_consistency_index: "proportion_consistency_index",
  morphometric_harmony_index: "morphometric_harmony_index",
  reference_deviation_index: "reference_deviation_index",
  top_contributing_features: "top_contributing_features",
  limitations: "limitations",
};

/** @type {Record<string, HelpEntry>} */
const HELP_ENTRIES = {
  "section.context": {
    en: {
      short: "Shows which input, execution mode, and result image the current analysis is based on.",
      interpretation: "Use this block to verify that the UI is showing the same artifact that the backend actually analyzed.",
    },
    ru: {
      short: "Показывает, на каком входе, режиме выполнения и результирующем изображении основан текущий анализ.",
      interpretation: "Этот блок нужен, чтобы проверить, что UI показывает тот же артефакт, который backend реально анализировал.",
    },
  },
  "section.analysis_request": {
    en: {
      short: "Controls how the backend will run the current analysis request and which reference group it will use.",
      interpretation: "These settings affect comparison context and runtime behavior, not the underlying morphometric scope.",
    },
    ru: {
      short: "Управляет тем, как backend выполнит текущий анализ и какую reference-группу он будет использовать.",
      interpretation: "Эти настройки влияют на контекст сравнения и режим выполнения, а не на сам morphometric scope.",
    },
  },
  "section.quality": {
    en: {
      short: "Summarizes face detection, pose, preprocessing quality gates, and any warnings before interpreting measurements.",
      interpretation: "If quality did not pass, treat downstream measurements and reference comparison more cautiously.",
    },
    ru: {
      short: "Сводка по детекции лица, pose, quality gate и предупреждениям перед интерпретацией измерений.",
      interpretation: "Если quality gate не пройден, к последующим измерениям и сравнению с reference нужно относиться осторожнее.",
    },
  },
  "section.visual_analysis": {
    en: {
      short: "Displays the image actually used for analysis together with landmarks, measurement lines, and symmetry guides.",
      interpretation: "The overlay is informative only for elements that can be derived from real preprocessing coordinates.",
    },
    ru: {
      short: "Показывает изображение, реально использованное в анализе, вместе с landmarks, линиями измерений и ориентирами симметрии.",
      interpretation: "Overlay информативен только для элементов, которые можно построить по реальным координатам preprocessing.",
    },
  },
  "section.raw_measurements": {
    en: {
      short: "Pixel-level distances measured directly from the current face geometry before normalization.",
      interpretation: "These values depend on crop scale and image resolution, so compare them mainly within the same sample.",
    },
    ru: {
      short: "Пиксельные расстояния, измеренные напрямую по текущей геометрии лица до нормализации.",
      interpretation: "Эти значения зависят от масштаба crop и разрешения изображения, поэтому в основном их стоит сравнивать внутри одного снимка.",
    },
  },
  "section.normalized_ratios": {
    en: {
      short: "Scale-independent ratios derived from raw measurements for more stable comparison across images.",
      interpretation: "These ratios are a main input to reference comparison because they are less sensitive to image size.",
    },
    ru: {
      short: "Масштабно-независимые отношения, полученные из сырых измерений для более стабильного сравнения между изображениями.",
      interpretation: "Эти отношения служат одним из основных входов для reference comparison, потому что меньше зависят от размера изображения.",
    },
  },
  "section.symmetry_metrics": {
    en: {
      short: "Describes left-right balance using distances around the eyes, nose, mouth, and center axis.",
      interpretation: "Higher symmetry-oriented values usually indicate lower measured asymmetry, not any subjective attractiveness judgment.",
    },
    ru: {
      short: "Описывает левую/правую сбалансированность по расстояниям вокруг глаз, носа, рта и центральной оси.",
      interpretation: "Более высокие значения симметрии обычно означают меньшую измеренную асимметрию, а не какую-либо субъективную привлекательность.",
    },
  },
  "section.angle_metrics": {
    en: {
      short: "Contains geometric angles and pose-related measures that help explain head orientation and line direction.",
      interpretation: "Large absolute pose angles can reduce trust in morphometric interpretation if the face is not well normalized.",
    },
    ru: {
      short: "Содержит геометрические углы и pose-метрики, которые объясняют ориентацию головы и направление линий.",
      interpretation: "Большие по модулю pose-углы могут снижать доверие к морфометрической интерпретации, если лицо плохо нормализовано.",
    },
  },
  "section.reference_comparison": {
    en: {
      short: "Compares the current normalized feature vector to a selected reference group using descriptive statistics.",
      interpretation: "This is a statistical deviation view, not a score of quality, beauty, or desirability.",
    },
    ru: {
      short: "Сравнивает текущий нормализованный вектор признаков с выбранной reference-группой через описательные статистики.",
      interpretation: "Это статистическое представление отклонений, а не оценка качества, красоты или желательности.",
    },
  },
  "section.feature_deviations": {
    en: {
      short: "Highlights which individual features contribute most strongly to deviation from the selected reference group.",
      interpretation: "Use this list to understand what drives the difference, not as a ranking of good or bad traits.",
    },
    ru: {
      short: "Подсвечивает признаки, которые сильнее всего вносят вклад в отклонение от выбранной reference-группы.",
      interpretation: "Этот список нужен, чтобы понять, что именно даёт разницу, а не как рейтинг «хороших» или «плохих» черт.",
    },
  },
  "section.composite_indices": {
    en: {
      short: "Optional aggregate summaries built from objective morphometric quantities already shown elsewhere in the report.",
      interpretation: "Treat them as compact summaries, not as standalone verdicts.",
    },
    ru: {
      short: "Опциональные агрегированные сводки, построенные из уже показанных в отчёте объективных морфометрических величин.",
      interpretation: "Их стоит воспринимать как компактные summary, а не как самостоятельный вердикт.",
    },
  },
  "section.limitations": {
    en: {
      short: "Lists reasons why interpretation may be less reliable for the current sample or reference context.",
      interpretation: "Read this block before drawing conclusions from outlier values.",
    },
    ru: {
      short: "Перечисляет причины, по которым интерпретация может быть менее надёжной для текущего снимка или reference-контекста.",
      interpretation: "Прочитайте этот блок перед тем, как делать выводы по outlier-значениям.",
    },
  },
  "section.artifacts": {
    en: {
      short: "Shows which files were generated by the backend and which image artifact was actually used for analysis and overlay.",
    },
    ru: {
      short: "Показывает, какие файлы создал backend и какой именно артефакт изображения был реально использован для анализа и overlay.",
    },
  },
  "field.reference_group_selector": {
    en: {
      short: "Selects the reference subgroup used for z-scores, percentiles, and Mahalanobis distance.",
      interpretation: "A more specific group can give a more relevant comparison, but only if it has enough samples.",
      warning: "If the exact group is sparse, fallback to a broader group may be more stable.",
    },
    ru: {
      short: "Выбирает reference-подгруппу для z-score, percentiles и Mahalanobis distance.",
      interpretation: "Более специфичная группа может дать более релевантное сравнение, но только если в ней достаточно примеров.",
      warning: "Если точная группа слишком мала, более широкая fallback-группа может быть устойчивее.",
    },
  },
  "field.manual_group_override": {
    en: {
      short: "Lets you override automatic or dropdown-based group selection with an explicit backend group string.",
      warning: "A malformed string can lead to missing reference-group matches.",
    },
    ru: {
      short: "Позволяет переопределить автоматический или выпадающий выбор группы явной строкой в backend-формате.",
      warning: "Некорректная строка может привести к отсутствию совпадения с reference-группой.",
    },
  },
  "field.execution_provider": {
    en: {
      short: "Chooses whether preprocessing runs on CPU or on CUDA GPU, if the backend runtime is ready.",
      interpretation: "This affects speed and runtime dependencies, not the morphometric meaning of the report.",
    },
    ru: {
      short: "Выбирает, выполняется ли preprocessing на CPU или на CUDA GPU, если backend runtime готов.",
      interpretation: "Это влияет на скорость и runtime-зависимости, а не на морфометрический смысл отчёта.",
    },
  },
  "field.device_id": {
    en: {
      short: "Specifies which visible GPU device the backend should try to use in CUDA mode.",
    },
    ru: {
      short: "Указывает, какую видимую GPU-карту backend должен попытаться использовать в CUDA-режиме.",
    },
  },
  "field.output_dir": {
    en: {
      short: "Overrides the backend directory where reports and generated image artifacts are saved.",
    },
    ru: {
      short: "Переопределяет backend-директорию, куда сохраняются отчёты и сгенерированные артефакты изображений.",
    },
  },
  "field.strict_gpu": {
    en: {
      short: "Prevents silent fallback to CPU when CUDA was explicitly requested.",
      warning: "Enable this when you need to verify that the GPU path is really active.",
    },
    ru: {
      short: "Запрещает тихий fallback на CPU, если CUDA была запрошена явно.",
      warning: "Включайте это, когда нужно убедиться, что GPU-путь реально активен.",
    },
  },
  processing_status: {
    en: {
      short: "Shows whether preprocessing finished successfully before metrics were computed.",
      interpretation: "A processed status means the pipeline reached the measurement stage; it does not guarantee quality passed.",
    },
    ru: {
      short: "Показывает, завершился ли preprocessing успешно до вычисления метрик.",
      interpretation: "Статус processed означает, что pipeline дошёл до этапа измерений, но это не гарантирует прохождение quality gate.",
    },
  },
  quality_pass: {
    en: {
      short: "Indicates whether the final quality gate accepted the image for reliable morphometric interpretation.",
      interpretation: "If this is false, some outputs may still be shown, but trust them less.",
    },
    ru: {
      short: "Показывает, принял ли финальный quality gate изображение как достаточно надёжное для морфометрической интерпретации.",
      interpretation: "Если значение false, некоторые результаты всё равно могут отображаться, но доверять им нужно меньше.",
    },
  },
  quality_fail_reason: {
    en: {
      short: "Explains which rule or combination of rules caused the quality gate to fail.",
      interpretation: "Use this together with pose and blur to understand whether the issue is fixable by a better input photo.",
    },
    ru: {
      short: "Объясняет, какое правило или комбинация правил привела к провалу quality gate.",
      interpretation: "Смотрите это вместе с pose и blur, чтобы понять, исправима ли проблема более качественным исходным фото.",
    },
  },
  processing_error_message: {
    en: {
      short: "Backend-side preprocessing error text, shown only when part of the pipeline failed.",
    },
    ru: {
      short: "Текст ошибки preprocessing со стороны backend, показывается только если часть pipeline завершилась ошибкой.",
    },
  },
  det_score: {
    en: {
      short: "Detector confidence for the selected face hypothesis.",
      interpretation: "Higher is usually better, but confidence alone does not guarantee the face is suitable for measurement.",
    },
    ru: {
      short: "Уверенность detector-а для выбранной гипотезы лица.",
      interpretation: "Большее значение обычно лучше, но сама по себе уверенность не гарантирует пригодность лица для измерений.",
    },
  },
  pose_yaw_deg: {
    en: {
      short: "Left-right head rotation angle.",
      interpretation: "Values near zero indicate a more frontal face. Larger absolute values can reduce comparability.",
    },
    ru: {
      short: "Угол поворота головы влево-вправо.",
      interpretation: "Значения около нуля означают более фронтальное лицо. Большие значения по модулю могут снижать сопоставимость.",
    },
  },
  pose_pitch_deg: {
    en: {
      short: "Up-down head tilt angle.",
      interpretation: "Large absolute pitch changes the perceived vertical geometry and can hurt measurement stability.",
    },
    ru: {
      short: "Угол наклона головы вверх-вниз.",
      interpretation: "Большой pitch по модулю меняет воспринимаемую вертикальную геометрию и может ухудшать стабильность измерений.",
    },
  },
  pose_roll_deg: {
    en: {
      short: "Head roll angle, or in-plane tilt.",
      interpretation: "Large roll can often be improved by auto-alignment before analysis.",
    },
    ru: {
      short: "Угол roll, то есть наклон головы в плоскости кадра.",
      interpretation: "Большой roll часто можно частично исправить auto-alignment перед анализом.",
    },
  },
  num_faces: {
    en: {
      short: "How many faces the detector found in the image.",
      warning: "The main 2D morphometric path expects a single clear face.",
    },
    ru: {
      short: "Сколько лиц detector нашёл на изображении.",
      warning: "Основной 2D morphometric path ожидает одно чёткое лицо.",
    },
  },
  blur_variance: {
    en: {
      short: "A blur proxy based on image sharpness variance.",
      interpretation: "Lower values usually mean the image is blurrier and facial geometry is less trustworthy.",
    },
    ru: {
      short: "Прокси размытия, основанная на дисперсии резкости изображения.",
      interpretation: "Более низкие значения обычно означают более размытое изображение и менее надёжную геометрию лица.",
    },
  },
  occlusion_score: {
    en: {
      short: "Optional occlusion estimate when the detector provides one.",
      warning: "Not every backend model exposes this value; absence does not mean no occlusion.",
    },
    ru: {
      short: "Опциональная оценка окклюзии, если detector её предоставляет.",
      warning: "Не каждая backend-модель возвращает это значение; отсутствие значения не означает отсутствие окклюзии.",
    },
  },
  raw_face_width_px: {
    en: { short: "Estimated face width in pixels from the detected face region." },
    ru: { short: "Оценка ширины лица в пикселях по обнаруженной области лица." },
  },
  raw_face_height_px: {
    en: { short: "Estimated face height in pixels from the detected face region." },
    ru: { short: "Оценка высоты лица в пикселях по обнаруженной области лица." },
  },
  raw_eye_distance_px: {
    en: { short: "Distance between the two eye anchor points in pixels." },
    ru: { short: "Расстояние между двумя опорными точками глаз в пикселях." },
  },
  raw_mouth_width_px: {
    en: { short: "Distance between the left and right mouth corner anchor points in pixels." },
    ru: { short: "Расстояние между левым и правым уголками рта в пикселях." },
  },
  raw_eye_to_nose_px: {
    en: { short: "Distance from eye center to nose anchor in pixels." },
    ru: { short: "Расстояние от центра глаз до опорной точки носа в пикселях." },
  },
  raw_nose_to_mouth_px: {
    en: { short: "Distance from nose anchor to mouth center in pixels." },
    ru: { short: "Расстояние от опорной точки носа до центра рта в пикселях." },
  },
  raw_eye_to_mouth_px: {
    en: { short: "Distance from eye center to mouth center in pixels." },
    ru: { short: "Расстояние от центра глаз до центра рта в пикселях." },
  },
  raw_nose_center_offset_px: {
    en: { short: "Horizontal offset of the nose anchor from the current face center axis." },
    ru: { short: "Горизонтальное смещение носа относительно текущей центральной оси лица." },
  },
  raw_mouth_center_offset_px: {
    en: { short: "Horizontal offset of the mouth center from the current face center axis." },
    ru: { short: "Горизонтальное смещение центра рта относительно текущей центральной оси лица." },
  },
  ratio_face_aspect: {
    en: {
      short: "Width-to-height relationship of the detected face region.",
      interpretation: "This is a neutral geometric proportion, not a value judgment.",
    },
    ru: {
      short: "Отношение ширины к высоте обнаруженной области лица.",
      interpretation: "Это нейтральная геометрическая пропорция, а не оценочное суждение.",
    },
  },
  ratio_eye_distance_to_mouth_width: {
    en: {
      short: "Compares eye spacing to mouth width using normalized geometry.",
    },
    ru: {
      short: "Сравнивает расстояние между глазами с шириной рта по нормализованной геометрии.",
    },
  },
  symmetry_index: {
    en: {
      short: "Compact summary of measured left-right balance across several symmetry-related features.",
      interpretation: "Higher values usually mean lower measured asymmetry.",
      warning: "This is not a beauty score; it only summarizes geometric balance.",
    },
    ru: {
      short: "Компактная сводка измеренного лево-правого баланса по нескольким symmetry-признакам.",
      interpretation: "Более высокие значения обычно означают меньшую измеренную асимметрию.",
      warning: "Это не beauty score; показатель лишь суммирует геометрический баланс.",
    },
  },
  sym_eye_nose_delta_ratio: {
    en: { short: "Normalized difference between left-eye-to-nose and right-eye-to-nose distances." },
    ru: { short: "Нормализованная разница между расстояниями левый глаз-нос и правый глаз-нос." },
  },
  sym_mouth_nose_delta_ratio: {
    en: { short: "Normalized difference between left-mouth-to-nose and right-mouth-to-nose distances." },
    ru: { short: "Нормализованная разница между расстояниями левый угол рта-нос и правый угол рта-нос." },
  },
  angle_eye_line_deg: {
    en: {
      short: "Angle of the eye line relative to the horizontal axis.",
      interpretation: "After successful normalization, this is usually close to zero for frontal images.",
    },
    ru: {
      short: "Угол линии глаз относительно горизонтальной оси.",
      interpretation: "После успешной нормализации для фронтальных изображений обычно близок к нулю.",
    },
  },
  angle_mouth_line_deg: {
    en: { short: "Angle of the mouth-corner line relative to the horizontal axis." },
    ru: { short: "Угол линии уголков рта относительно горизонтальной оси." },
  },
  angle_eye_center_to_mouth_center_deg: {
    en: { short: "Direction angle from eye center to mouth center." },
    ru: { short: "Направляющий угол от центра глаз к центру рта." },
  },
  angle_nose_to_mouth_center_deg: {
    en: { short: "Direction angle from nose anchor to mouth center." },
    ru: { short: "Направляющий угол от опорной точки носа к центру рта." },
  },
  reference_group: {
    en: {
      short: "The subgroup of the reference model used for comparison.",
      interpretation: "Reference statistics depend on this group definition and its sample size.",
    },
    ru: {
      short: "Подгруппа reference-модели, используемая для сравнения.",
      interpretation: "Reference-статистики зависят от определения этой группы и её размера.",
    },
  },
  sample_count: {
    en: {
      short: "How many samples contributed to the selected reference group.",
      interpretation: "Larger groups usually provide more stable summary statistics.",
      warning: "Small sample counts can make percentiles and standard deviations less reliable.",
    },
    ru: {
      short: "Сколько примеров вошло в выбранную reference-группу.",
      interpretation: "Более крупные группы обычно дают более стабильные summary-статистики.",
      warning: "Малый размер группы может сделать percentiles и стандартные отклонения менее надёжными.",
    },
  },
  complete_case_count: {
    en: {
      short: "How many samples had the full feature set required for the stored reference statistics.",
    },
    ru: {
      short: "Сколько примеров имели полный набор признаков, необходимый для сохранённых reference-статистик.",
    },
  },
  mahalanobis_distance: {
    en: {
      short: "Multivariate statistical distance from the selected reference group.",
      interpretation: "Higher values mean the combined feature vector is farther from the reference center.",
      warning: "This is a deviation measure, not a beauty score or a ranking.",
    },
    ru: {
      short: "Многомерное статистическое расстояние до выбранной reference-группы.",
      interpretation: "Большие значения означают, что совокупный вектор признаков дальше от центра reference-распределения.",
      warning: "Это мера отклонения, а не beauty score и не рейтинг.",
    },
  },
  percentile_position: {
    en: {
      short: "Approximate location of a feature within the stored reference distribution.",
      interpretation: "Percentiles are estimated from saved control points and are less precise for sparse groups.",
    },
    ru: {
      short: "Приблизительное положение признака внутри сохранённого reference-распределения.",
      interpretation: "Percentiles оцениваются по сохранённым контрольным точкам и менее точны для малых групп.",
    },
  },
  mean_absolute_z_score: {
    en: {
      short: "Average absolute standardized deviation across available compared features.",
      interpretation: "Larger values mean the feature set is, on average, farther from the selected reference means.",
    },
    ru: {
      short: "Среднее абсолютное стандартизованное отклонение по доступным сравниваемым признакам.",
      interpretation: "Большие значения означают, что набор признаков в среднем дальше от средних выбранной reference-группы.",
    },
  },
  proportion_consistency_index: {
    en: {
      short: "Compact summary of how internally consistent the normalized facial ratios are.",
      interpretation: "Higher values mean lower spread among the selected normalized proportions.",
    },
    ru: {
      short: "Компактная сводка того, насколько согласованы между собой нормализованные пропорции лица.",
      interpretation: "Более высокие значения означают меньший разброс среди выбранных нормализованных пропорций.",
    },
  },
  morphometric_harmony_index: {
    en: {
      short: "Optional blend of symmetry, proportion consistency, and reference deviation summaries.",
      interpretation: "Use it as a compact descriptor only; inspect the underlying metrics for real interpretation.",
    },
    ru: {
      short: "Опциональное объединение сводок по симметрии, согласованности пропорций и reference deviation.",
      interpretation: "Используйте его только как компактный дескриптор; для реальной интерпретации смотрите исходные метрики.",
    },
  },
  reference_deviation_index: {
    en: {
      short: "Inverse-style summary derived from the multivariate reference deviation.",
      interpretation: "Higher values indicate a smaller overall deviation from the selected reference group.",
    },
    ru: {
      short: "Инверсная сводка, построенная из многомерного reference deviation.",
      interpretation: "Более высокие значения означают меньшее суммарное отклонение от выбранной reference-группы.",
    },
  },
  top_contributing_features: {
    en: {
      short: "The features with the largest absolute standardized deviation from the selected reference group.",
      interpretation: "These features explain where the current sample differs most strongly from the reference distribution.",
    },
    ru: {
      short: "Признаки с наибольшим абсолютным стандартизованным отклонением от выбранной reference-группы.",
      interpretation: "Эти признаки объясняют, где текущий снимок сильнее всего отличается от reference-распределения.",
    },
  },
  limitations: {
    en: {
      short: "Important caveats that limit the confidence of the current analysis or comparison.",
    },
    ru: {
      short: "Важные оговорки, которые ограничивают уверенность в текущем анализе или сравнении.",
    },
  },
  "toggle.bbox": {
    en: { short: "Show or hide the detected face bounding box." },
    ru: { short: "Показать или скрыть рамку обнаруженного лица." },
  },
  "toggle.landmarks": {
    en: { short: "Show or hide the five anchor facial points used by the overlay." },
    ru: { short: "Показать или скрыть пять опорных facial points, используемых overlay." },
  },
  "toggle.raw_measurements": {
    en: { short: "Show or hide the main measurement lines used for raw distances." },
    ru: { short: "Показать или скрыть основные линии измерений для сырых расстояний." },
  },
  "toggle.eyes": {
    en: { short: "Shows eye-region points and lines such as interpupillary, intercanthal, and eye-width guides." },
    ru: { short: "Показывает точки и линии глазной зоны: межзрачковую, межкантальную и ширину глаз." },
  },
  "toggle.nose": {
    en: { short: "Shows nose anchors, width segments, base points, and nose-axis geometry." },
    ru: { short: "Показывает носовые опорные точки, ширины, основание носа и ось носа." },
  },
  "toggle.mouth": {
    en: { short: "Shows mouth corners, mouth width, philtrum-related lines, and lower-midface links." },
    ru: { short: "Показывает уголки рта, ширину рта, линии филтрума и связи средней/нижней части лица." },
  },
  "toggle.chin_jaw": {
    en: { short: "Shows chin, jaw, and cheekbone proxy geometry where reliable 2D anchors are available." },
    ru: { short: "Показывает геометрию подбородка, челюсти и прокси-скул там, где есть надёжные 2D-точки." },
  },
  "toggle.thirds": {
    en: { short: "Shows vertical thirds and horizontal guides used for upper, middle, and lower face proportion proxies." },
    ru: { short: "Показывает вертикальные трети и горизонтальные направляющие для прокси верхней, средней и нижней трети лица." },
  },
  "toggle.symmetry": {
    en: { short: "Show or hide the face axis and left-right asymmetry guides." },
    ru: { short: "Показать или скрыть ось лица и ориентиры лево-правой асимметрии." },
  },
  "toggle.angles": {
    en: { short: "Show or hide angle-related guide lines for pose and line orientation." },
    ru: { short: "Показать или скрыть направляющие линии для угловых метрик и ориентации pose." },
  },
  "toggle.debug_mode": {
    en: { short: "Enables dense-landmark inspection tools for debugging landmark mapping and coordinate-space issues." },
    ru: { short: "Включает инструменты просмотра dense-landmarks для отладки mapping и coordinate-space ошибок." },
  },
  "toggle.all_landmarks": {
    en: { short: "Shows the full dense landmark set returned by preprocessing, not only the named overlay anchors." },
    ru: { short: "Показывает полный dense-набор landmarks из preprocessing, а не только именованные опорные точки overlay." },
  },
  "toggle.debug_indices": {
    en: { short: "Shows numeric indices for dense landmarks to help verify landmark mapping." },
    ru: { short: "Показывает числовые индексы dense-landmarks для проверки корректности mapping-схемы." },
  },
  "toggle.labels": {
    en: { short: "Show or hide inline labels with distance, angle, and ratio values on the overlay." },
    ru: { short: "Показать или скрыть подписи со значениями расстояний, углов и отношений прямо на overlay." },
  },
};

function normalizeKey(key) {
  return String(key ?? "").trim().toLowerCase();
}

/**
 * @param {string | undefined | null} key
 * @returns {string | null}
 */
export function resolveHelpKey(key) {
  if (!key) {
    return null;
  }
  const normalized = normalizeKey(key);
  return HELP_ALIASES[normalized] || HELP_ALIASES[String(key)] || (HELP_ENTRIES[normalized] ? normalized : null);
}

/**
 * @param {string | undefined | null} key
 * @param {Locale} locale
 * @returns {LocalizedHelpEntry | null}
 */
export function getHelpEntry(key, locale) {
  const resolvedKey = resolveHelpKey(key);
  if (!resolvedKey) {
    return null;
  }
  const entry = HELP_ENTRIES[resolvedKey];
  if (!entry) {
    return null;
  }
  return entry[locale] || null;
}

/**
 * @param {string | undefined | null} key
 * @param {Locale} locale
 * @param {string | undefined | null} fallbackLabel
 * @returns {string}
 */
export function getHelpLabel(key, locale, fallbackLabel) {
  const entry = getHelpEntry(key, locale);
  if (entry?.label) {
    return entry.label;
  }
  return fallbackLabel ? String(fallbackLabel) : String(key ?? "");
}
