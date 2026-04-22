# Полный перечень запросов для Codex по шагам

Ниже — готовые copy-paste запросы. Отправляй **по одному запросу за раз**, строго по порядку. Следующий шаг запускай только после того, как предыдущий успешно завершён и проверен.

---

## 0. Общие правила для всех запусков Codex

Используй этот блок как префикс или просто держи его как правило для каждого шага:

```text
Работай только в рамках текущего шага. Не перескакивай к следующим фазам. Не реализуй весь проект сразу. Не переписывай уже готовые части без крайней необходимости. Все пути и параметры выноси в конфиг. Не хардкодь абсолютные пути. После выполнения верни: 1) список изменённых/созданных файлов, 2) команды запуска/проверки, 3) краткий отчёт о результате, 4) подтверждение готовности к следующему шагу.
```

---

## 1. Фаза 0 — Bootstrap проекта

```text
Открой файл codex_step_by_step_plan/01_PHASE_00_BOOTSTRAP.md и выполни только эту фазу.

Нужно подготовить каркас проекта для локального ML-пайплайна facial aesthetics scoring. Реализуй только bootstrap:
- создай структуру каталогов:
  - configs/
  - data/raw, data/interim, data/processed, data/features
  - scripts/
  - src/face_aesthetics/
  - reports/
  - tests/
- добавь README.md, pyproject.toml или requirements.txt, .gitignore, .env.example
- настрой базовые зависимости:
  numpy, pandas, pyarrow, scikit-learn, torch, torchvision, opencv-python, insightface, onnxruntime, xgboost, lightgbm, matplotlib, fastapi, uvicorn, pydantic, pyyaml, hydra-core
- создай конфиги:
  - configs/base.yaml
  - configs/data.yaml
  - configs/preprocess.yaml
  - configs/train_baseline.yaml
  - configs/train_fusion.yaml
  - configs/infer.yaml
- создай пустые каркасы модулей:
  - src/face_aesthetics/core/
  - src/face_aesthetics/data/
  - src/face_aesthetics/features/
  - src/face_aesthetics/models/
  - src/face_aesthetics/pipelines/
  - src/face_aesthetics/services/
  - src/face_aesthetics/utils/
- добавь простой smoke script, который читает конфиг, печатает пути и не падает на import

Важно:
- не реализуй preprocessing
- не пиши training logic
- не используй notebook как основной способ запуска

После выполнения верни:
1) список созданных файлов,
2) команды запуска,
3) подтверждение, что структура готова к следующей фазе.
```

---

## 2. Фаза 1 — Dataset manifest

```text
Открой файл codex_step_by_step_plan/02_PHASE_01_DATASET_MANIFEST.md и выполни только эту фазу.

Нужно реализовать воспроизводимый dataset manifest для SCUT-FBP5500, размещённого локально в data/raw/.

Сделай:
- реализуй scripts/prepare_dataset.py
- собери manifest в CSV и Parquet
- для каждого изображения сохрани:
  - image_id
  - image_path
  - subject_id, если есть
  - split (train/val/test), если задаётся явно
  - score_mean
  - score_std, если доступен
  - дополнительные метаданные, если есть
- проверь:
  - файл существует
  - изображение читается
  - нет дубликатов image_id
  - нет пустых путей
- сохрани:
  - data/interim/dataset_manifest.parquet
  - data/interim/dataset_manifest.csv
  - reports/metrics/dataset_manifest_summary.json

Желательная структура кода:
- src/face_aesthetics/data/manifest.py
- src/face_aesthetics/data/io.py
- src/face_aesthetics/utils/filesystem.py

Важно:
- не извлекай признаки
- не запускай training
- не делай face detection на этом шаге

После выполнения верни:
1) путь к manifest,
2) размер train/val/test,
3) количество пропусков и битых файлов,
4) команды для следующей фазы.
```

---

## 3. Фаза 2 — Preprocessing pipeline

```text
Открой файл codex_step_by_step_plan/03_PHASE_02_PREPROCESSING.md и выполни только эту фазу.

Нужно реализовать воспроизводимый preprocessing pipeline:
- детекция лица
- alignment
- quality gate
- pose estimation
- сохранение промежуточных артефактов

Сделай:
- реализуй scripts/extract_preprocessing_artifacts.py
- подключи готовый detector/alignment, приоритетно InsightFace
- для каждого изображения из manifest:
  - найди лицо
  - проверь, что лицо одно
  - сделай aligned crop
  - сохрани bbox
  - сохрани landmarks
  - сохрани confidence detector-а
- подключи оценку pose:
  - yaw / pitch / roll
- введи quality gate:
  - нет лица
  - больше одного лица
  - слишком маленький bbox
  - слишком сильный blur
  - слишком экстремальный pose
  - сильная окклюзия, если есть надёжный признак
- сохрани результаты:
  - aligned images в data/processed/aligned/
  - preprocessing manifest в data/processed/preprocessed_manifest.parquet
- для каждой записи храни:
  - face_detected
  - num_faces
  - bbox
  - landmarks
  - det_score
  - yaw, pitch, roll
  - quality_pass
  - quality_fail_reason
  - aligned_image_path

Важно:
- не извлекай deep embeddings в этой фазе
- не обучай модели
- не смешивай preprocessing и feature extraction в одном скрипте

После выполнения верни:
1) количество успешно обработанных изображений,
2) число отклонённых примеров по каждой причине,
3) 3–5 примеров сохранённых aligned crops,
4) путь к preprocessed_manifest.parquet.
```

---

## 4. Фаза 3 — Feature extraction

```text
Открой файл codex_step_by_step_plan/04_PHASE_03_FEATURE_EXTRACTION.md и выполни только эту фазу.

Нужно извлечь и сохранить признаки так, чтобы обучение можно было проводить без повторного запуска тяжёлого CV pipeline.

Сделай:
- реализуй scripts/extract_features.py
- на вход бери только записи с quality_pass = true
- извлеки группы признаков:

A. Deep embeddings
- используй готовый face encoder
- получи фиксированный embedding-вектор
- сохрани размерность и имя encoder-а

B. Geometry features
- на основе landmarks вычисли:
  - межглазное расстояние
  - ширина/высота лица
  - отношения nose/mouth/eye distances
  - symmetry proxies
  - jaw / cheekbone / forehead ratios, если landmarks позволяют

C. Pose features
- yaw
- pitch
- roll

D. Quality features
- sharpness / blur proxy
- brightness / contrast proxy
- face box area ratio

Сохрани:
- data/features/deep_embeddings.parquet
- data/features/geometry_features.parquet
- data/features/fusion_features.parquet

Обеспечь единый ключ image_id.
Ожидаемая структура колонок:
- image_id
- target_score_mean
- target_score_std
- split
- emb_*
- geom_*
- pose_*
- quality_*

Важно:
- не обучай модели
- не храни всё в одном giant JSON
- не дублируй признаки без нужды

После выполнения верни:
1) размерность deep embedding,
2) число geometry features,
3) число итоговых fusion features,
4) пути к parquet-файлам.
```

---

## 5. Фаза 4 — Baseline на deep embeddings

```text
Открой файл codex_step_by_step_plan/05_PHASE_04_DEEP_BASELINE.md и выполни только эту фазу.

Нужно построить первый сильный baseline только на готовых deep embeddings.

Сделай:
- реализуй scripts/train_baseline.py
- используй как вход data/features/deep_embeddings.parquet
- обучи минимум 2 baseline-модели:
  - XGBoostRegressor или LightGBMRegressor
  - MLPRegressor или небольшой PyTorch MLP
- посчитай метрики:
  - MAE
  - RMSE
  - Spearman
  - Pearson
- сохрани:
  - модель
  - конфиг обучения
  - metrics JSON
  - prediction file для validation/test
  - feature importance для табличной модели, если доступно
- положи артефакты в:
  - reports/metrics/
  - models/baselines/ или аналогичную папку

Важно:
- не добавляй geometry features
- не делай fusion
- не смешивай ranking и regression

После выполнения верни:
1) лучшую baseline-модель,
2) метрики на val/test,
3) список сохранённых артефактов,
4) короткий вывод: baseline готов для сравнения с geometry/fusion.
```

---

## 6. Фаза 5 — Geometry-only baseline

```text
Открой файл codex_step_by_step_plan/06_PHASE_05_GEOMETRY_BASELINE.md и выполни только эту фазу.

Нужно проверить, сколько сигнала дают только геометрические признаки лица без deep embeddings.

Сделай:
- реализуй scripts/train_geometry.py
- используй как вход data/features/geometry_features.parquet
- обучи:
  - линейный baseline: Ridge или ElasticNet
  - деревообразный baseline: XGBoostRegressor или LightGBMRegressor
- посчитай те же метрики, что и в deep baseline:
  - MAE
  - RMSE
  - Spearman
  - Pearson
- сравни результаты с фазой deep baseline
- сохрани отчёт reports/metrics/geometry_vs_deep_baseline.json

Важно:
- не добавляй embeddings
- не обучай fusion head в этой фазе

После выполнения верни:
1) метрики geometry-only,
2) сравнение с deep-only,
3) вывод: geometry branch полезен или почти не даёт выигрыша.
```

---

## 7. Фаза 6 — Fusion model

```text
Открой файл codex_step_by_step_plan/07_PHASE_06_FUSION_MODEL.md и выполни только эту фазу.

Нужно объединить deep, geometry, pose и quality признаки и получить наиболее сильную регрессионную модель.

Сделай:
- реализуй scripts/train_fusion.py
- используй data/features/fusion_features.parquet
- обучи минимум 2 варианта:
  - tabular model: XGBoostRegressor или LightGBMRegressor
  - небольшой MLP на fusion-векторе
- при наличии смысла реализуй simple ensemble:
  - среднее или weighted average лучших предсказаний
- посчитай метрики:
  - MAE
  - RMSE
  - Spearman
  - Pearson
- построй сравнение:
  - deep-only
  - geometry-only
  - fusion
- сохрани:
  - лучшую модель
  - predictions
  - comparison report
  - графики: scatter, residuals, pred-vs-target

Важно:
- не добавляй ranking loss
- не смешивай distribution modeling

После выполнения верни:
1) лучшую fusion-модель,
2) сводную таблицу сравнения,
3) вывод, что считать текущим production-like baseline.
```

---

## 8. Фаза 7 — Pairwise ranking model

```text
Открой файл codex_step_by_step_plan/08_PHASE_07_RANKING_MODEL.md и выполни только эту фазу.

Нужно проверить, даёт ли ranking-постановка преимущество над прямой регрессией на субъективной шкале.

Сделай:
- реализуй scripts/train_ranking.py
- сгенерируй пары (image_a, image_b, label), где label = 1, если score_a > score_b
- используй признаки:
  - deep-only или
  - fusion-вектор
- реализуй один из вариантов:
  - pairwise logistic loss
  - margin ranking loss
  - Bradley-Terry style scoring
- посчитай метрики:
  - pairwise accuracy
  - Spearman после восстановления scalar score, если это поддерживается архитектурой
- сравни ranking с лучшей regression-моделью

Важно:
- не усложняй Siamese-архитектуру без необходимости
- не тащи тяжёлые backbone-обучения

После выполнения верни:
1) ranking metric,
2) сравнение с regression,
3) решение: ranking оставлять как дополнительную ветку или нет.
```

---

## 9. Фаза 8 — Uncertainty / distribution modeling

```text
Открой файл codex_step_by_step_plan/09_PHASE_08_UNCERTAINTY_MODEL.md и выполни только эту фазу.

Нужно научить модель предсказывать не только средний score, но и неопределённость или разброс оценок.

Сделай:
- реализуй scripts/train_uncertainty.py
- используй:
  - score_mean
  - score_std или распределение оценок, если оно доступно
- поддержи один из вариантов:
  - mean + variance regression
  - histogram/bin classification
  - Gaussian NLL
- посчитай метрики:
  - MAE для mean
  - calibration-related metric
  - correlation between predicted uncertainty and annotator disagreement
- сохрани:
  - predictions с mean/uncertainty
  - calibration plots
  - uncertainty report

Важно:
- не усложняй до mixture density nets без причины
- не ломай существующий regression pipeline

После выполнения верни:
1) модель uncertainty,
2) метрики mean + uncertainty,
3) краткий вывод, полезна ли неопределённость в проекте.
```

---

## 10. Фаза 9 — Robustness evaluation

```text
Открой файл codex_step_by_step_plan/10_PHASE_09_ROBUSTNESS_EVAL.md и выполни только эту фазу.

Нужно проверить устойчивость лучшей модели к небольшим изменениям входного изображения.

Сделай:
- реализуй scripts/evaluate_robustness.py
- для набора test-изображений примени мягкие преобразования:
  - небольшой crop shift
  - brightness change
  - contrast change
  - jpeg compression
  - small yaw-like perturbation, если реализуемо
  - blur
- для каждого изображения оцени:
  - исходный pred
  - pred после perturbation
  - абсолютное отклонение
- построй:
  - summary table
  - boxplot / histogram дрейфа предсказаний
  - список самых нестабильных примеров

Важно:
- не переобучай модель на augmented data на этом шаге
- только оценка устойчивости

После выполнения верни:
1) средний drift,
2) worst-case примеры,
3) вывод, насколько модель устойчива к nuisance-факторам.
```

---

## 11. Фаза 10 — Subgroup analysis / bias diagnostics

```text
Открой файл codex_step_by_step_plan/11_PHASE_10_SUBGROUP_ANALYSIS.md и выполни только эту фазу.

Нужно проверить, одинаково ли ведёт себя модель на разных подгруппах.

Сделай:
- реализуй scripts/evaluate_subgroups.py
- если доступны subgroup labels, посчитай метрики раздельно по:
  - gender
  - age buckets
  - ethnicity / race buckets
  - другим доступным группам
- для каждой подгруппы вычисли:
  - MAE
  - Spearman
  - mean prediction
  - residual statistics
- сохрани:
  - subgroup report
  - таблицы
  - визуализации разницы метрик

Важно:
- не добавляй subgroup labels как признаки в scorer без явной причины
- это диагностическая, а не оптимизационная фаза

После выполнения верни:
1) таблицу метрик по подгруппам,
2) краткий вывод о смещениях,
3) рекомендации, надо ли дополнительно калибровать модель.
```

---

## 12. Фаза 11 — Inference pipeline

```text
Открой файл codex_step_by_step_plan/12_PHASE_11_INFERENCE_PIPELINE.md и выполни только эту фазу.

Нужно собрать единый CLI pipeline для одного изображения:
image -> face detection -> alignment -> features -> score

Сделай:
- реализуй scripts/infer_image.py
- на вход принимай:
  - путь к изображению
  - путь к конфигу
  - путь к модели
- pipeline должен:
  - прочитать изображение
  - найти лицо
  - проверить quality gate
  - сделать alignment
  - извлечь признаки
  - загрузить лучшую модель
  - выдать score
  - опционально выдать uncertainty
- сформируй человекочитаемый результат:
  - JSON
  - консольный вывод
- сохраняй inference log

Важно:
- не делай веб-сервис на этом шаге
- сначала CLI, потом API

После выполнения верни:
1) пример команды инференса,
2) пример JSON-ответа,
3) список обрабатываемых ошибок.
```

---

## 13. Фаза 12 — API

```text
Открой файл codex_step_by_step_plan/13_PHASE_12_API.md и выполни только эту фазу.

Нужно обернуть inference pipeline в локальный API.

Сделай:
- реализуй FastAPI сервис
- добавь endpoint:
  - POST /score
- на вход принимай:
  - изображение
  - optional flags для debug mode
- на выход возвращай:
  - score
  - uncertainty, если поддерживается
  - quality_pass
  - failure_reason, если инференс отклонён
  - технические метаданные в debug mode
- добавь:
  - startup loading model
  - health endpoint
  - простую обработку исключений

Важно:
- не добавляй auth, БД и очереди
- это локальный исследовательский API, а не production backend

После выполнения верни:
1) список endpoint-ов,
2) пример запроса,
3) пример ответа.
```

---

## 14. Финальный проход — логирование, тесты, отчёты, README

```text
Открой файл codex_step_by_step_plan/14_LOGGING_TESTING_REPORTS.md и выполни финальный quality pass по уже реализованному проекту.

Нужно:
- добавить единый logging layer для:
  - dataset preparation
  - preprocessing
  - feature extraction
  - training
  - evaluation
  - inference
- минимально логировать:
  - timestamp
  - phase
  - input path
  - output artifact path
  - counts processed/skipped/failed
  - key metrics
- добавить тесты минимум на:
  - manifest builder не создаёт дубликаты
  - preprocessing pipeline корректно обрабатывает "нет лица"
  - feature extractor возвращает ожидаемую размерность
  - training script читает признаки и выдаёт метрики
  - inference CLI не падает на валидном изображении
- убедиться, что в reports/ появляются:
  - dataset summary
  - preprocessing summary
  - baseline metrics
  - fusion comparison
  - robustness report
  - subgroup analysis report
  - inference examples
- обновить финальный README:
  - структура проекта
  - команды установки
  - порядок фаз
  - команды запуска
  - где искать артефакты
  - как выполнить inference

Важно:
- это не новая фаза моделирования, а финальный quality pass
- не ломай рабочие части проекта
- исправляй только реально нужные места

После выполнения верни:
1) список добавленных тестов,
2) список логируемых событий/полей,
3) список отчётов,
4) финальные команды для полного воспроизведения проекта.
```

---

## Как отправлять

Оптимальный порядок такой:
1. Сначала отправляешь запрос №1.
2. Проверяешь результат и файлы.
3. Только потом отправляешь №2.
4. И так далее до №14.

Если хочешь жёстче контролировать Codex, перед каждым запросом добавляй одну строку:

```text
Не переходи к следующей фазе, пока полностью не завершишь текущую и не вернёшь артефакты и команды проверки.
```
