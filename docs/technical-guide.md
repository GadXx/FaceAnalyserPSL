# FaceAnalyserPSL

Локальный ML/CV-пайплайн для **objective morphometric face analysis** и
отдельного **experimental learned scorer**.
Legacy-имя репозитория `FaceAnalyserPSL` и пакет `src.face_aesthetics`
сохранены для обратной совместимости импортов и entry points.

Проект имеет две независимые части:
- objective morphometric pipeline (измерения, ratios, symmetry, angles, optional 3D)
- experimental scorer layer (subjective score, обучаемый по labeled dataset)

## Scope

Objective часть возвращает только морфометрические признаки и метаданные качества.
Experimental scorer не смешивается с objective report и возвращается отдельным блоком API/UI.

Scorer policy:
- primary score = continuous regression prediction
- nearest group = только справочная привязка к обучающим папкам
- classifier/pairwise/ordinal = auxiliary diagnostics
- confidence + warnings обязательны для интерпретации
- scorer geometry-only by default (без raw image pixels и metadata leakage)

## Supported Datasets

Текущий manifest builder поддерживает:
- `fairface`
- `utkface`
- `chicago_face_database`
- `scut_compat`
- generic image-folder adapter

Manifest собирается в unified schema и сохраняется в:
- `data/interim/dataset_manifest.parquet`
- `data/interim/dataset_manifest.csv`

Reference layer for scorer:
- `FairFace` и `UTKFace` используются как population/reference baseline
- beauty-labeled dataset (папки `1..9.5`) используется только как target for scoring
- reference-relative features (`ref_z_*`, `ref_dir_*`, `ref_ratio_*`, `ref_absdev_*`, `ref_percentile_*`)
  считаются по reference stats, не по beauty dataset

## Expected Folder Structure

Минимальная ожидаемая структура:

```text
configs/
data/
  raw/
    fairface/
    utkface/
    chicago_face_database/
  interim/
    insightface_models/
      models/
        buffalo_l/
    3d_models/
  processed/
    aligned_faces/
    3d_artifacts/
    preprocessed_manifest.parquet/
  features/
models/
  reference/
reports/
  examples/
  metrics/
scripts/
src/
tests/
```

## Setup

Рекомендуемый порядок установки с пустого repo state:

1. Создать и активировать virtualenv.
2. Установить Python dependencies.
3. Положить raw datasets в `data/raw/`.
4. Положить InsightFace 2D models в `data/interim/insightface_models/models/buffalo_l/`.
5. При необходимости включить optional 3D branch.
6. Собрать manifest.
7. Запустить preprocessing.
8. Запустить feature extraction.
9. Построить reference model.
10. Запустить single-image inference или API.

Установка зависимостей:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MediaPipe backend for dense landmarks is included in the main dependency list:

```bash
python -m pip install -r requirements.txt
```

Если `mediapipe` недоступен в текущем окружении, pipeline не падает:
- InsightFace-only extraction продолжает работать
- `reports/metrics/landmark_capability_report.json` фиксирует `mediapipe_available=false`
- `mediapipe_unavailable_reason` содержит причину import/runtime failure

Для `FaceLandmarker` также нужен model asset:
- `data/interim/mediapipe/face_landmarker.task`

MediaPipe delegate policy:
- `delegate=cpu|gpu|auto`
- `fallback_to_cpu=true|false`
- diagnostics пишутся в `reports/metrics/mediapipe_runtime_diagnostics.json`

Важно:
- `InsightFace` ускоряется через `ONNXRuntime CUDAExecutionProvider`
- `MediaPipe` ускоряется через свой `GPU delegate`, а не через CUDAExecutionProvider
- если в системе renderer показывает `llvmpipe`, то фактически используется software/OpenGL fallback, а не реальный GPU

Быстрый smoke-check импортов и конфигов:

```bash
python -m src.face_aesthetics.core.smoke_check
```

## Dependency Checklist

### 2D Baseline

Для dataset manifest external model assets не нужны.

Для 2D preprocessing / feature extraction / inference обязательны:
- `data/interim/insightface_models/models/buffalo_l/det_10g.onnx`
- `data/interim/insightface_models/models/buffalo_l/2d106det.onnx`

Дополнительно для auxiliary deep embeddings:
- `data/interim/insightface_models/models/buffalo_l/w600k_r50.onnx`

Reference-aware inference требует ещё:
- `models/reference/reference_model.joblib`
- `models/reference/reference_group_statistics.parquet`

### Optional 3D Path

Optional 3D branch использует **template-based 3D lifting**, а не learned 3D
reconstruction.

Для неё нужны:
- те же 2D InsightFace models, что и для baseline
- `data/interim/3d_models/canonical_face_5pt.npy`

Формат template:
- `numpy` array формы `(5, 3)`
- порядок точек:
  - `left_eye`
  - `right_eye`
  - `nose`
  - `left_mouth`
  - `right_mouth`

Если включены:
- `preprocess.three_d.enabled = true`
- `preprocess.three_d.auto_create_default_template = true`

и файл отсутствует, репозиторий автоматически создаёт deterministic default
template в:
- `data/interim/3d_models/canonical_face_5pt.npy`

## Dataset Manifest

Сбор unified manifest:

```bash
.venv/bin/python scripts/prepare_dataset.py
```

Артефакты:
- `data/interim/dataset_manifest.parquet`
- `data/interim/dataset_manifest.csv`
- `reports/metrics/dataset_manifest_summary.json`

## Preprocessing

Preprocessing pipeline выполняет:
- face detection
- landmarks
- alignment
- pose estimation
- quality gate
- optional 3D artifact generation

Запуск:

```bash
.venv/bin/python scripts/extract_preprocessing_artifacts.py
```

Основные outputs:
- `data/processed/preprocessed_manifest.parquet`
- `data/processed/aligned_faces/`
- `reports/metrics/preprocessing_summary.json`

GPU readiness check:

```bash
.venv/bin/python scripts/extract_preprocessing_artifacts.py \
  --gpu-readiness-check \
  --preferred-execution-provider CUDAExecutionProvider \
  --device-id 0
```

Если нужен strict GPU mode:

```bash
.venv/bin/python scripts/extract_preprocessing_artifacts.py \
  --preferred-execution-provider CUDAExecutionProvider \
  --device-id 0 \
  --strict-gpu true
```

## Feature Extraction

Feature extraction использует только eligible preprocessing rows и сохраняет:
- `data/features/measurements_raw.parquet`
- `data/features/measurements_normalized.parquet`
- `data/features/symmetry_features.parquet`
- `data/features/geometry_features.parquet`
- optional `data/features/geometry_3d_features.parquet`
- optional `data/features/deep_embeddings.parquet`
- `reports/metrics/feature_registry.json`
- `reports/metrics/feature_audit.json`

Запуск:

```bash
.venv/bin/python scripts/extract_features.py
```

`feature_registry.json` хранит прозрачный справочник признаков:
- `feature_name`
- `feature_family`
- `status`
- `alias_of`
- `formula`
- `notes`

Статусы используются для audit слоя:
- `keep`
- `new`
- `deprecated_but_kept_for_compatibility`
- `redundant`
- `unstable`

`feature_audit.json` фиксирует:
- общее число реализованных geometry features
- распределение по статусам
- compatibility aliases и явные дубликаты
- признаки, пока оставленные как `unstable`

Multi-source landmark capability artifacts:
- `reports/metrics/landmark_capability_report.json`
- optional `reports/debug/mediapipe_landmarks.json`
- optional `reports/debug/mediapipe_landmarks_indices.png`
- optional `reports/debug/mediapipe_landmark_regions.json`

MediaPipe path остаётся optional:
- при наличии dependency может вернуть dense `468/478` landmarks
- добавляет отдельные `mediapipe_*` geometry features
- не заменяет InsightFace anchors, а расширяет feature space
- scorer по умолчанию не использует proxy/raw MediaPipe features без reliability audit

Текущий расширенный 2D feature space включает:
- vertical proportions / thirds proxies
- horizontal proportions for eye, nose, mouth, jaw, cheekbone proxies
- midline and asymmetry offsets
- eye-zone metrics
- nose / philtrum / lip / chin metrics
- additional contour and axis angles

## Reference Model

`scripts/train_baseline.py` сохраняет legacy-name entry point, но теперь строит
не regressor baseline, а **reference model** по morphometric features.
Это групповые reference-статистики для сравнения признаков, а не supervised
модель субъективной оценки лица. Готовые артефакты уже находятся в
`models/reference/`; обучение нужно только при пересборке reference-набора.

Reference groups выбираются конфигом:
- `group_label_aliases`
- `group_by`
- `allowed_reference_splits`
- `minimum_group_sample_count`
- `minimum_complete_case_count`

Запуск:

```bash
.venv/bin/python scripts/train_baseline.py
```

Outputs:
- `models/reference/reference_model.joblib`
- `models/reference/reference_group_statistics.parquet`
- `models/reference/reference_model_config_snapshot.json`
- `reports/metrics/reference_model_summary.json`

## Single-Sample Inference

Для одиночного изображения:

```bash
.venv/bin/python scripts/run_inference.py \
  --image-path path/to/image.jpg \
  --group "sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=White"
```

Если `--group` не передан, inference пытается прочитать sidecar metadata рядом с
изображением:
- `image.jpg.json`
- `image.json`
- `image.jpg.metadata.json`
- `image.metadata.json`

Inference pipeline делает:
- preprocessing и quality gate
- intermediate full-frame normalization into `analysis_image`
- normalized face crop into `aligned_face` when alignment/crop succeeds
- raw measurements
- normalized ratios
- symmetry / angle metrics
- reference lookup
- z-score / Mahalanobis / percentile comparison
- JSON и Markdown report

Outputs:
- `reports/examples/analysis_image.jpg`
- `reports/examples/aligned_face.jpg`
- `reports/examples/annotated_analysis.jpg`
- `reports/examples/sample_report.json`
- `reports/examples/sample_report.md`
- `reports/metrics/reporting_summary.json`

Роли image artifacts:
- `original_image_path`: исходное входное изображение
- `analysis_image_path`: промежуточный full-frame image после rotation/normalization
- `aligned_image_path`: normalized face crop
- `measurement_image_path`: фактическое изображение, по которому считаются downstream measurements
- `overlay_image_path`: фактическое изображение, поверх которого рисуется overlay
- `annotated_image_path`: сохранённый overlay result

Текущая single-image inference логика:
- если redetection на `aligned_face` успешен, downstream measurements/ratios/symmetry/angles/reference comparison
  считаются в coordinate space `aligned_face`
- если redetection на `aligned_face` неуспешен, controlled fallback остаётся на `analysis_image`

## API

FastAPI service:

```bash
.venv/bin/python scripts/run_api.py --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `GET /reference-groups`
- `POST /analyze`
- `POST /analyze-upload`

Минимальный example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "data/raw/fairface/train/3.jpg",
    "group": "sex_group_normalized=Female,age_group_normalized=3-9,ethnicity_group_normalized=Black",
    "preferred_execution_provider": "CPUExecutionProvider"
  }'
```

API переиспользует тот же service layer, что и CLI inference. Dependency errors
возвращаются явно, без тихого fallback.

File upload example:

```bash
curl -X POST http://127.0.0.1:8000/analyze-upload \
  -F "file=@data/raw/fairface/train/3.jpg" \
  -F "group=sex_group_normalized=Female,age_group_normalized=3-9,ethnicity_group_normalized=Black" \
  -F "preferred_execution_provider=CPUExecutionProvider"
```

Для `POST /analyze-upload` нужен `python-multipart`. Upload flow сохраняет файл во
внутреннюю runtime директорию и затем переиспользует тот же inference service,
что и path-based CLI/API сценарий.

## Web Frontend

Отдельное Vue 3 приложение лежит в:

- `apps/web`

Оно использует:
- `GET /health`
- `GET /reference-groups`
- `POST /analyze-upload`

Фронтенд также визуализирует morphometric overlay поверх загруженного фото по
реальным координатам из API:
- `input_quality.detection.bbox`
- `visualization.image_size`
- `visualization.anchors`
- `visualization.named_points`
- `visualization.derived_points`
- `visualization.lines`
- `visualization.axes`
- `visualization.guides`

Запуск:

```bash
cd apps/web
cp .env.example .env
npm install
npm run dev
```

Нужная env-переменная:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Frontend поддерживает:
- drag-and-drop upload
- image preview и очистку выбранного файла
- выбор reference group или manual override
- выбор execution provider
- structured UI для morphometric report без raw-JSON-only представления

Подробнее:
- `apps/web/README.md`

## Robustness Evaluation

```bash
.venv/bin/python scripts/evaluate_robustness.py \
  --image-path path/to/image.jpg \
  --group "sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=White"
```

Outputs:
- `reports/metrics/robustness_summary.json`
- `reports/metrics/robustness_feature_deltas.parquet`

## Subgroup Validation

```bash
.venv/bin/python scripts/validate_subgroups.py
```

Outputs:
- `reports/metrics/subgroup_validation.json`
- `reports/metrics/reference_group_coverage.json`
- `reports/metrics/subgroup_coverage.parquet`

Это transparency/coverage analysis, а не fairness marketing claim.

## Optional 3D Branch

3D branch остаётся optional и не влияет на базовый 2D pipeline.

Включение:
- `configs/preprocess.yaml` -> `preprocess.three_d.enabled`
- `configs/features.yaml` -> `features.three_d.enabled`

Что делает текущая 3D ветка:
- берёт 2D landmarks и pose (`yaw/pitch/roll`) из 2D preprocessing
- применяет canonical 5-point face template
- делает deterministic pose-aware lifting
- сохраняет `landmarks_3d`
- считает `geom3d_*` признаки

Чего она не делает:
- не является learned depth model
- не является DECA / 3DDFA_V2 / 3DMM fitting pipeline
- не обязательна для baseline/reference path

3D preprocessing outputs:
- `data/processed/3d_artifacts/`
- `data/processed/3d_artifacts/manifest.parquet`
- `reports/metrics/preprocessing_3d_summary.json`

3D feature output:
- `data/features/geometry_3d_features.parquet`

3D smoke test:

```bash
.venv/bin/python scripts/extract_preprocessing_artifacts.py \
  --limit 16 \
  --batch-size 8 \
  --overwrite true \
  --resume false \
  --preferred-execution-provider CPUExecutionProvider

.venv/bin/python scripts/extract_features.py
```

## Tests

Полный test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Текущие тесты покрывают:
- dataset adapters
- preprocessing pipeline smoke/failure paths
- measurement formulas
- reference model construction
- optional 3D branch
- robustness evaluation
- subgroup validation
- CLI smoke path
- API smoke path

## Limitations

- Проект не вычисляет attractiveness / beauty / PSL style outputs.
- Reference comparison зависит от качества reference groups и их sample counts.
- GPU availability зависит от локального ONNX Runtime / CUDA runtime состояния.
- Optional 3D branch сейчас template-based и intentionally simpler than learned
  3D reconstruction.
- Single-image inference path currently remains 2D-oriented; optional 3D branch
  протестирован на preprocessing + feature extraction path.

## Experimental Scoring Layer

В репозитории есть отдельный experimental scorer, который не вмешивается в
objective reference model. Он обучается на dataset tree, где имя папки
является score label, например:

- `dataset/5/`
- `dataset/5.5/`
- `dataset/7/`
- `dataset/8.5/`

Для scorer не нужно обучать собственный детектор лица. Feature extraction
использует настроенные landmark assets (включая
`data/interim/mediapipe/face_landmarker.task`), а обучение scorer строит
tabular-модели по геометрическим признакам. Команды ниже перезаписывают
артефакты в `models/scoring/`; сохраните копию текущих файлов, если хотите
оставить исходный scorer.

Что важно:

- scorer учится на размеченных примерах и не предполагает, что каждая
  отдельная метрика имеет универсальное "идеальное" значение.
- Высокий score может получаться из разных комбинаций сильных и средних
  признаков; модель учит эти сочетания статистически по dataset, а не по
  вручную заданной таблице идеалов.
- В strict-scale режиме `5` трактуется как ordinary / mass baseline zone,
  `9.5` как elite anchor уровня top-tier examples, `10` как почти
  недостижимая theoretical upper bound, а `0` как reserved invalid / severe
  anomaly bound.
- `predicted_score` является continuous output primary supervised regression
  model и не должен подменяться ближайшей обучающей группой.
- `display_score` это только UI-округление continuous `predicted_score`,
  например:
  - `8.6079 -> 8.61`
- Сам `predicted_score` может быть непрерывным:
  - `7.32`
  - `7.86`
  - `9.02`
- Пользовательская semantic шкала scorer остаётся `0..10`, даже если в текущем
  dataset отсутствуют крайние labels `0` и `10`.
- При этом модель обучается только на реально наблюдаемых labels, а report
  отдельно хранит `observed_training_label_range`, чтобы не маскировать
  отсутствие support у крайних значений.
- На текущем dataset labels покрывают промежуточный диапазон вроде `1..9.5`;
  это не ошибка, но крайние области `0..1` и `9.5..10` поддерживаются хуже.
- `nearest_score_group` это ближайшая обучающая группа, а не сама оценка.
  Она используется только как справочная привязка к папкам датасета.
- `predicted_group` из classification baseline теперь выводится отдельно от
  regression блока и не должен интерпретироваться как замена
  `nearest_score_group`.
- scorer не обучается по raw image pixels напрямую: фото используется только
  для извлечения morphometric metrics, а модели получают только tabular
  geometry-derived features.
- Этот слой субъективный/экспериментальный и должен использоваться отдельно от
  baseline morphometric report.
- По умолчанию scorer обучается только на `quality_pass=true` строках.
- По умолчанию из scorer training исключены raw pixel features (`raw_*`,
  `*_px`), чтобы уменьшить crop-scale bias и leakage от абсолютного масштаба.
- По умолчанию scorer работает в режиме `geometry_only`: используются только
  normalized ratios, symmetry metrics, angle metrics и pose-related geometry
  features. Metadata, file paths и embeddings не попадают в train.
- Model selection теперь идёт через cross-validation, а не через один
  validation split.
- Scoring manifest теперь делится на `train / val / test`.
- Pairwise ranking строится только после image-level split:
  - train pairs только из train images
  - validation/test metrics считаются на val/test images без пересечения с
    train images
- Final score нужно интерпретировать вместе с `confidence`; low-confidence
  output не должен считаться надёжным.
- Primary score теперь отдаётся из лучшей learned regression model.
- Primary model selection теперь дополнительно штрафует regression-to-mean:
  учитываются low/high group bias, elite-group underprediction, group macro MAE
  и calibration slope, а не только global MAE.
- `raw_predicted_score` и `calibrated_predicted_score` хранятся отдельно.
  Calibration fit делается только на validation predictions и не использует
  test split для обучения.
- Pairwise ranking, ordinal scorer, classifier baseline и ensemble остаются
  auxiliary diagnostics и confidence signals. Они не подменяют primary score,
  если не подтверждают качество на holdout.
- Pairwise ranking особенно полезен для маленького субъективного dataset tree:
  он учится сравнивать пары по относительному порядку score bins, а не только
  регрессировать абсолютное значение.
- Local explanations и component contributions показывают, какие признаки и
  зоны лица сдвинули learned score вверх или вниз относительно baseline модели.
  Это объяснение learned model, а не объективная истина о "качестве" частей
  лица.

Подготовка scoring manifest:

```bash
.venv/bin/python scripts/prepare_scoring_dataset.py --dataset-root ./dataset
```

Feature extraction для scoring dataset:

```bash
.venv/bin/python scripts/extract_scoring_features.py
```

Обучение continuous scorer:

```bash
.venv/bin/python scripts/train_scorer.py
```

Single-image scorer inference:

```bash
.venv/bin/python scripts/run_scorer.py --image-path path/to/photo.jpg
```

Single-image scorer explanation:

```bash
.venv/bin/python scripts/explain_scorer.py --image-path path/to/photo.jpg
```

API scorer upload endpoint:

```bash
POST /score-upload
```

The web UI calls this endpoint after the main objective analysis upload. The
scoring result is shown in a separate "Experimental Learned Score" section and
does not replace the objective morphometric report.

The objective analysis response also includes an explicit `expanded_geometry`
section (MediaPipe-derived groups: face oval, eyes, brows, nose, mouth/lips,
cheekbones/midface, jaw/chin) with `source` and `reliability` tags
(`reliable`/`proxy`/`unstable`) for each feature.

The UI scorer section now shows:

- primary score from the learned regression model
- confidence and warning reasons
- local explanation with top positive / negative feature contributions
- global learned feature/component weights (dataset-learned importance)
- component contributions by facial region / feature family
- auxiliary diagnostics for pairwise, ordinal, classifier, and ensemble models

Important:

- component contributions are learned explanations, not objective sub-scores
- global weights and local contributions are different concepts:
  global weight = model-wide importance, local contribution = per-image score shift
- classifier predicted group is diagnostic only and is not shown as the primary score
- low-confidence results should not be interpreted as reliable

Основные scoring artifacts:

- `data/scoring/scoring_manifest.parquet`
- `data/scoring/scoring_features.parquet`
- `data/scoring/scoring_pairs.parquet`
- `models/scoring/scorer_regressor.joblib`
- `models/scoring/scorer_classifier.joblib`
- `models/scoring/pairwise_ranker.joblib`
- `models/scoring/ordinal_scorer.joblib`
- `reports/metrics/scorer_training_summary.json`
- `reports/metrics/scorer_cv_results.json`
- `reports/metrics/scorer_model_selection.json`
- `reports/metrics/scorer_selected_features.json`
- `reports/metrics/scoring_pairwise_summary.json`
- `reports/metrics/pairwise_ranker_summary.json`
- `reports/metrics/ordinal_scorer_summary.json`
- `reports/metrics/scorer_ensemble_summary.json`
- `reports/metrics/scorer_feature_importance.json`
- `reports/metrics/scorer_feature_correlations.json`
- `reports/metrics/scorer_global_feature_importance.json`
- `reports/metrics/scorer_component_importance.json`
- `reports/metrics/scorer_feature_effects.json`
- `reports/metrics/scorer_group_confusion.json`
- `reports/metrics/scorer_groupwise_errors.json`
- `reports/metrics/scorer_calibration_by_group.json`
- `reports/metrics/scorer_extreme_bias_summary.json`
- `reports/metrics/scorer_prediction_examples_by_group.json`
- `reports/metrics/scoring_quality_failures.json`
- `reports/metrics/expanded_geometry_scorer_comparison.json`
- `reports/metrics/scorer_weighted_model_comparison.json`
- `reports/metrics/scorer_learned_feature_weights.json`
- `reports/metrics/scorer_learned_component_weights.json`
- `reports/metrics/scorer_component_weight_summary.json`
- `reports/metrics/scorer_component_balanced_feature_selection.json`
- `reports/examples/scorer_report.json`
- `reports/examples/scorer_report.md`

Ограничения scoring dataset:

- scorer не создаёт synthetic labels между папками; промежуточные значения
  появляются только как output regression model;
- extrapolation вне observed score range не гарантируется;
- в текущем dataset могут отсутствовать `0` и `10`; пользовательская шкала при
  этом всё равно остаётся `0..10`, но predictions near unsupported extremes
  должны интерпретироваться как lower-confidence;
- если dataset сейчас покрывает, например, только `1..9.5`, то предсказания
  около `0..1` и `9.5..10` должны интерпретироваться как low-confidence;
- если score group имеет мало samples, training summary явно помечает её как
  weak-support group;
- quality-failed images сохраняются в scoring features со status, но по
  умолчанию не используются в train.
- pairwise / ordinal / ensemble метрики без независимого image-level holdout
  могут быть завышены; поэтому текущая схема использует отдельный test split.
- web UI показывает score отдельно от objective analysis и явно помечает его
  как experimental subjective layer.
- качество score сильно зависит от размера, чистоты и согласованности dataset
  tree; маленькие и дисбалансные группы быстро снижают надёжность.
- при маленьком dataset tree regression и classification могут расходиться; это
  отражается в `confidence.reasons` и warning-блоке scorer report.
- component contributions и feature effects dataset-dependent: они отражают
  learned relationships в текущем scorer, а не causal truth и не objective
  morphometric norm.
- если хочешь улучшить scorer quality, сначала увеличивай число изображений на
  группу и выравнивай support по score bins, а не только меняй модель.

## Runbook

Короткий runbook от пустого repo state до первого успешного 2D inference:

1. Склонировать repo.
2. Создать `.venv` и установить `requirements.txt`.
3. Положить raw dataset image в `data/raw/...` или подготовить dataset tree.
4. Положить InsightFace files:
   - `data/interim/insightface_models/models/buffalo_l/det_10g.onnx`
   - `data/interim/insightface_models/models/buffalo_l/2d106det.onnx`
5. Собрать dataset manifest:
   - `.venv/bin/python scripts/prepare_dataset.py`
6. Запустить preprocessing:
   - `.venv/bin/python scripts/extract_preprocessing_artifacts.py --preferred-execution-provider CPUExecutionProvider`
7. Запустить feature extraction:
   - `.venv/bin/python scripts/extract_features.py`
8. Построить reference model:
   - `.venv/bin/python scripts/train_baseline.py`
9. Выполнить single-image inference:
   - `.venv/bin/python scripts/run_inference.py --image-path path/to/image.jpg --group "sex_group_normalized=...,age_group_normalized=...,ethnicity_group_normalized=..."`
10. Проверить outputs:
   - `reports/examples/sample_report.json`
   - `reports/examples/sample_report.md`

Это и есть первый успешный 2D morphometric inference path.
