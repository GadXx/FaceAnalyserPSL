# Фаза 6 — Fusion model

## Цель

Объединить deep, geometry, pose и quality признаки и получить наиболее сильную регрессионную модель.

## Зависимости

Требуются завершённые:
- `Фаза 4`
- `Фаза 5`

## Что должен сделать Codex

1. Реализовать `scripts/train_fusion.py`.
2. Использовать:
   - `data/features/fusion_features.parquet`

3. Обучить как минимум 2 варианта:
   - tabular model (`XGBoostRegressor` или `LightGBMRegressor`);
   - небольшой `MLP` на fusion-векторе.

4. При наличии смысла реализовать simple ensemble:
   - среднее или weighted average лучших предсказаний.

5. Посчитать метрики:
   - MAE
   - RMSE
   - Spearman
   - Pearson

6. Построить сравнение:
   - deep-only
   - geometry-only
   - fusion

7. Сохранить:
   - лучшую модель;
   - predictions;
   - comparison report;
   - графики: scatter, residuals, pred-vs-target.

## Критерии завершения

- fusion-модель обучена;
- есть сравнение со всеми baseline;
- выбрана текущая best model.

## Команды проверки

```bash
python scripts/train_fusion.py
```

## Ограничения

- не добавлять ranking loss в этой фазе;
- не смешивать distribution modeling.

## Что должен вернуть Codex после выполнения

1. лучшую fusion-модель;
2. сводную таблицу сравнения;
3. вывод, что считать текущим production-like baseline.
