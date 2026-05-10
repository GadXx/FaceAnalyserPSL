# FaceAnalyserPSL

Локальный ML/CV-пайплайн для **objective morphometric face analysis**.
Legacy-имя репозитория `FaceAnalyserPSL` и пакет `src.face_aesthetics`
сохранены только для обратной совместимости импортов и entry points.

Проект **не** предназначен для beauty / attractiveness / PSL / appeal scoring.
Активные пайплайны вычисляют только:
- 2D facial measurements
- normalized morphometric ratios
- symmetry and angle metrics
- reference-based z-score / Mahalanobis / percentile comparisons
- optional template-based 3D geometry features

## Scope

Система возвращает только objective morphometric outputs и технические
метаданные качества. Никакие active code paths не должны вводить:
- `beauty_score`
- `attractiveness_score`
- `psl_score`
- `appeal_score`

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
