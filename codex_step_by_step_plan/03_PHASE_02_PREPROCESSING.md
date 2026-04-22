# Фаза 2 — Preprocessing pipeline

## Цель

Реализовать воспроизводимый preprocessing pipeline:
- детекция лица;
- alignment;
- quality gate;
- pose estimation;
- сохранение промежуточных артефактов.

## Зависимости

Требуются завершённые:
- `Фаза 0`
- `Фаза 1`

## Что должен сделать Codex

1. Реализовать `scripts/extract_preprocessing_artifacts.py` или аналогичный скрипт.
2. Подключить готовый detector/alignment:
   - приоритетно `InsightFace`
   - fallback: другой совместимый backend при явной необходимости

3. Для каждого изображения из manifest:
   - найти лицо;
   - убедиться, что лицо одно;
   - сделать aligned crop;
   - сохранить bbox;
   - сохранить landmarks;
   - сохранить confidence detector-а.

4. Подключить оценку pose:
   - yaw / pitch / roll;
   - можно через отдельную модель или служебный модуль.

5. Ввести quality gate:
   - нет лица;
   - больше одного лица;
   - слишком маленький bbox;
   - слишком сильный blur;
   - слишком экстремальный pose;
   - сильная окклюзия, если есть надёжный признак

6. Сохранять результаты:
   - aligned images в `data/processed/aligned/`
   - preprocessing manifest в `data/processed/preprocessed_manifest.parquet`

7. Для каждой записи хранить:
   - `face_detected`
   - `num_faces`
   - `bbox`
   - `landmarks`
   - `det_score`
   - `yaw`, `pitch`, `roll`
   - `quality_pass`
   - `quality_fail_reason`
   - `aligned_image_path`

## Критерии завершения

- preprocessing можно прогнать батчем;
- aligned faces сохраняются на диск;
- все отклонённые изображения имеют причину;
- есть итоговая статистика pass/fail.

## Команды проверки

```bash
python scripts/extract_preprocessing_artifacts.py
```

## Ограничения

- не извлекать deep embeddings в этой фазе;
- не обучать модели;
- не смешивать preprocessing и feature extraction в одном скрипте.

## Что должен вернуть Codex после выполнения

1. количество успешно обработанных изображений;
2. число отклонённых примеров по каждой причине;
3. 3–5 примеров сохранённых aligned crops;
4. путь к `preprocessed_manifest.parquet`.
