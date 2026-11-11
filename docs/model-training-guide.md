# Обучение собственного scorer

Этот гайд описывает, как обучить экспериментальную модель оценки по своим размеченным изображениям. Scorer учится на геометрических признаках, извлечённых из фото: normalized ratios, symmetry и angles. Он не обучает детектор лица и не анализирует сырые пиксели напрямую.

В репозитории уже есть готовые файлы `models/scoring/`, поэтому обучение не требуется, чтобы попробовать текущую версию. Нижеописанные команды создадут новый scorer и заменят эти локальные файлы.

## Что понадобится

- Python и зависимости из `requirements.txt`.
- MediaPipe FaceLandmarker model asset по пути `data/interim/mediapipe/face_landmarker.task`.
- Текущие reference features из `data/reference/reference_features.parquet` (файл включён в репозиторий), если используешь настройки reference-relative features по умолчанию.
- Собственный датасет с числовой оценкой для каждой фотографии.

Для scorer не надо устанавливать InsightFace ONNX assets и обучать собственный landmark detector: этот pipeline извлекает scoring features через настроенный MediaPipe backend. InsightFace assets нужны для основного morphometric inference pipeline.

## 1. Подготовь набор с оценками

Каждая папка верхнего уровня — одна числовая оценка. В папке могут быть изображения `.jpg`, `.jpeg`, `.png`, `.webp` или `.bmp`.

```text
dataset/
  4/
    photo-001.jpg
    photo-002.jpg
  5.5/
    photo-003.jpg
  7/
    photo-004.jpg
```

Названия папок должны преобразовываться в число: используй точку в дробных значениях (`5.5`, не `5,5`). Значения папок становятся target score без дополнительной нормализации. Можно использовать свою шкалу, но модель сможет надёжно предсказывать только области, представленные в данных.

Перед запуском:

- Примени одинаковую шкалу и критерии ко всем изображениям.
- По возможности выровняй количество фотографий между score-группами и включи примеры по всему нужному диапазону.
- Удали повреждённые изображения, дубликаты и почти одинаковые кадры.
- Не клади фотографии одного человека в разные почти дублирующие наборы: split в текущем скрипте делается по файлам, а не по личности. Повтор одного человека в разных split может завысить оценку качества.
- Получи согласие на обработку изображений и не добавляй личные фотографии в Git.

Скрипт делит изображения отдельно внутри каждой score-группы: примерно 70% в train, 15% в validation и 15% в test. При менее чем 5 снимках в группе проект выдаёт предупреждение о слабой поддержке; на практике для устойчивых метрик нужно существенно больше примеров на группу.

## 2. Подготовь окружение

Команды запускай из корня репозитория. Создай виртуальное окружение, если ещё не создавал:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Положи MediaPipe asset по пути `data/interim/mediapipe/face_landmarker.task`. Убедись, что файл `data/reference/reference_features.parquet` на месте. Перед обучением скопируй существующую папку `models/scoring/`, если хочешь сохранить включённые в репозиторий модели: `train_scorer.py` записывает новые артефакты по тем же путям.

Если датасет расположен не в `./dataset`, передай его путь в первой команде ниже. Путь может быть абсолютным.

## 3. Собери manifest и splits

```bash
python scripts/prepare_scoring_dataset.py --dataset-root ./dataset
```

Скрипт создаст:

- `data/scoring/scoring_manifest.parquet` — список файлов, оценок и назначенных split;
- `reports/metrics/scoring_dataset_summary.json` — число изображений и групп, размеры split и предупреждения о слабых группах.

Проверь summary перед feature extraction: число групп, число файлов в каждой группе и отсутствие пустых групп должны соответствовать твоему набору.

## 4. Извлеки признаки

```bash
python scripts/extract_scoring_features.py
```

Изображения обрабатываются MediaPipe; для каждой строки записываются геометрические признаки и статус quality gate. Результат сохраняется в:

- `data/scoring/scoring_features.parquet`;
- `reports/metrics/scoring_feature_summary.json`.

Проверь `feature_rows`, `quality_pass_rows` и `processed_rows`. Если почти все строки не прошли quality gate, сначала проверь наличие FaceLandmarker asset, качество изображений и сообщения в summary — обучение на малом числе прошедших строк не даст полезной модели.

## 5. Обучи модели

```bash
python scripts/train_scorer.py
```

Pipeline сравнит несколько регрессионных и вспомогательных моделей, выберет primary regressor по validation/CV метрикам, выполнит calibration на validation split и сохранит диагностику. Test split предназначен для итоговой оценки и не используется для обучения калибратора.

Основные выходы:

- `models/scoring/scorer_regressor.joblib` — primary regression model;
- `models/scoring/score_calibrator.joblib` — калибратор, если его удалось обучить;
- `models/scoring/scorer_classifier.joblib`, `pairwise_ranker.joblib`, `ordinal_scorer.joblib` — вспомогательные модели;
- `reports/metrics/scorer_training_summary.json`;
- `reports/metrics/scorer_cv_results.json`;
- `reports/metrics/scorer_model_selection.json`.

В конфигурации по умолчанию включён `geometry_only`: в training не передаются исходные пиксели, пути файлов и metadata. Артефакты с данными и метриками находятся в `data/scoring/` и `reports/metrics/`; эти каталоги игнорируются Git.

## 6. Оцени результат и проверь inference

Смотри на holdout/test метрики в training summary, особенно MAE и Spearman, затем проверь ошибки отдельно по score-группам в `reports/metrics/scorer_groupwise_errors.json`. Высокое качество на train или CV само по себе не доказывает, что модель обобщается на новые изображения. Проверь, что test примеры не являются почти дубликатами train изображений и что отдельные группы не представлены слишком слабо.

Запусти scorer на новом фото:

```bash
python scripts/run_scorer.py --image-path path/to/photo.jpg
```

Отчёты появятся в `reports/examples/scorer_report.json` и `reports/examples/scorer_report.md`. Учитывай `confidence` и warnings: score — предсказание модели, обученной на конкретных субъективных метках, а не объективная или универсальная характеристика лица.

## Отдельно: reference baseline

`scripts/train_baseline.py` обучает другую часть системы. Она строит reference-статистики морфометрических признаков по группам, а не scorer score. Для нового reference набора нужны изображения и метаданные в поддерживаемом формате внутри `data/raw/`; полный порядок `prepare_dataset.py` → preprocessing → `extract_features.py` → `train_baseline.py` приведён в [техническом руководстве](technical-guide.md#runbook).
