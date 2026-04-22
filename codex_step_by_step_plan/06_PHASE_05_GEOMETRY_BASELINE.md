# Фаза 5 — Geometry-only baseline

## Цель

Проверить, сколько сигнала дают только геометрические признаки лица без deep embeddings.

## Зависимости

Требуются завершённые:
- `Фаза 3`
- `Фаза 4`

## Что должен сделать Codex

1. Реализовать `scripts/train_geometry.py`.
2. Взять как вход:
   - `data/features/geometry_features.parquet`

3. Обучить:
   - линейный baseline (`Ridge` или `ElasticNet`);
   - деревообразный baseline (`XGBoostRegressor` или `LightGBMRegressor`).

4. Посчитать те же метрики, что и в deep baseline:
   - MAE
   - RMSE
   - Spearman
   - Pearson

5. Сравнить результаты с `Фаза 4`.

6. Сохранить отчёт:
   - `reports/metrics/geometry_vs_deep_baseline.json`

## Критерии завершения

- geometry baseline обучен;
- сравнение с deep baseline выполнено;
- есть вывод, стоит ли geometry branch добавлять во fusion.

## Команды проверки

```bash
python scripts/train_geometry.py
```

## Ограничения

- не добавлять embeddings;
- не обучать fusion head в этой фазе.

## Что должен вернуть Codex после выполнения

1. метрики geometry-only;
2. сравнение с deep-only;
3. вывод: geometry branch полезен или почти не даёт выигрыша.
