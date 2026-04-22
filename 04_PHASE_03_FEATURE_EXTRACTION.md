# Фаза 3 — Feature extraction

## Цель

Извлечь и сохранить признаки так, чтобы все эксперименты обучения можно было проводить без повторного запуска тяжёлого CV pipeline.

## Зависимости

Требуются завершённые:
- `Фаза 0`
- `Фаза 1`
- `Фаза 2`

## Что должен сделать Codex

1. Реализовать `scripts/extract_features.py`.
2. На вход брать только записи с `quality_pass = true`.
3. Извлечь следующие группы признаков:

### A. Deep embeddings
- использовать готовый face encoder;
- получить фиксированный embedding-вектор;
- сохранить размерность и имя encoder-а.

### B. Geometry features
На основе landmarks вычислить:
- межглазное расстояние;
- ширина/высота лица;
- отношения nose/mouth/eye distances;
- symmetry proxies;
- jaw / cheekbone / forehead ratios, если landmarks позволяют.

### C. Pose features
- yaw
- pitch
- roll

### D. Quality features
- sharpness / blur proxy
- brightness / contrast proxy
- face box area ratio

4. Сохранить:
   - `data/features/deep_embeddings.parquet`
   - `data/features/geometry_features.parquet`
   - `data/features/fusion_features.parquet`

5. Обеспечить единый ключ:
   - `image_id`

## Желательная структура колонок

- `image_id`
- `target_score_mean`
- `target_score_std`
- `split`
- `emb_*`
- `geom_*`
- `pose_*`
- `quality_*`

## Критерии завершения

- признаки извлекаются из preprocessed manifest;
- одна запись = один `image_id`;
- все таблицы можно объединить по ключу;
- нет утечки test в train.

## Команды проверки

```bash
python scripts/extract_features.py
```

## Ограничения

- не обучать модели в этой фазе;
- не хранить всё в одном gigantic JSON;
- не дублировать признаки без нужды.

## Что должен вернуть Codex после выполнения

1. размерность deep embedding;
2. число geometry features;
3. число итоговых fusion features;
4. пути к parquet-файлам.
