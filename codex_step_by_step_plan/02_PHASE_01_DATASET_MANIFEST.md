# Фаза 1 — Dataset manifest

## Цель

Сделать воспроизводимый индекс датасета, чтобы все следующие шаги работали не с "папкой картинок", а с нормальным manifest-файлом.

## Зависимости

Требуется завершённая `Фаза 0`.

## Вход

- локально размещённый датасет `SCUT-FBP5500` в `data/raw/`
- при необходимости: дополнительные метаданные и split-файлы

## Что должен сделать Codex

1. Реализовать `scripts/prepare_dataset.py`.
2. Пройти по датасету и собрать manifest в формате CSV или Parquet.
3. Для каждого изображения сохранить:
   - `image_id`
   - `image_path`
   - `subject_id` если есть
   - `split` (train/val/test), если задаётся явно
   - `score_mean`
   - `score_std` если доступен
   - дополнительные метаданные, если они есть в датасете

4. Проверить:
   - файл существует;
   - изображение читается;
   - нет дубликатов `image_id`;
   - нет пустых путей.

5. Сохранить:
   - `data/interim/dataset_manifest.parquet`
   - `data/interim/dataset_manifest.csv`
   - краткую статистику по датасету в `reports/metrics/dataset_manifest_summary.json`

## Рекомендуемая структура кода

- `src/face_aesthetics/data/manifest.py`
- `src/face_aesthetics/data/io.py`
- `src/face_aesthetics/utils/filesystem.py`

## Критерии завершения

- manifest создаётся одной командой;
- manifest воспроизводим;
- все изображения из manifest доступны по путям;
- есть базовая статистика.

## Команды проверки

```bash
python scripts/prepare_dataset.py
```

## Ограничения

- не извлекать признаки;
- не запускать training;
- не делать детекцию лиц на этом шаге.

## Что должен вернуть Codex после выполнения

1. путь к manifest;
2. размер train/val/test;
3. количество пропусков/битых файлов;
4. список команд для следующей фазы.
