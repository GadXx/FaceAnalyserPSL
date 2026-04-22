# Фаза 4 — Baseline на deep embeddings

## Цель

Построить первый сильный baseline только на готовых deep embeddings.

## Зависимости

Требуются завершённые:
- `Фаза 0`
- `Фаза 1`
- `Фаза 2`
- `Фаза 3`

## Что должен сделать Codex

1. Реализовать `scripts/train_baseline.py`.
2. Взять как вход:
   - `data/features/deep_embeddings.parquet`

3. Обучить как минимум 2 baseline-модели:
   - `XGBoostRegressor` или `LightGBMRegressor`
   - `MLPRegressor` или небольшой PyTorch MLP

4. Метрики:
   - MAE
   - RMSE
   - Spearman
   - Pearson

5. Сохранить:
   - модель;
   - конфиг обучения;
   - metrics JSON;
   - prediction file для validation/test;
   - feature importance для табличной модели, если доступно.

6. Положить артефакты в:
   - `reports/metrics/`
   - `models/baselines/` или аналогичную папку

## Критерии завершения

- baseline обучается одной командой;
- метрики считаются автоматически;
- результаты сохраняются в файлы;
- есть воспроизводимый split.

## Команды проверки

```bash
python scripts/train_baseline.py
```

## Ограничения

- не добавлять geometry features;
- не делать fusion;
- не смешивать ranking и regression.

## Что должен вернуть Codex после выполнения

1. лучшую baseline-модель;
2. метрики на val/test;
3. список сохранённых артефактов;
4. короткий вывод: baseline готов для сравнения с geometry/fusion.
