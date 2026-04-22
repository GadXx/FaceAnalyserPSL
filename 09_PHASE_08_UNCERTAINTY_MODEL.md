# Фаза 8 — Uncertainty / distribution modeling

## Цель

Научить модель предсказывать не только средний score, но и неопределённость или разброс оценок.

## Зависимости

Требуются завершённые:
- `Фаза 6`
- желательно `Фаза 7`, но не обязательно

## Что должен сделать Codex

1. Реализовать `scripts/train_uncertainty.py`.
2. Использовать:
   - `score_mean`
   - `score_std` или распределение оценок, если оно доступно

3. Поддержать один из вариантов:
   - mean + variance regression;
   - histogram/bin classification;
   - Gaussian NLL.

4. Метрики:
   - MAE для mean;
   - calibration-related metric;
   - correlation between predicted uncertainty and annotator disagreement.

5. Сохранить:
   - predictions с mean/uncertainty;
   - calibration plots;
   - uncertainty report.

## Критерии завершения

- uncertainty-предсказание работает;
- есть хотя бы базовая калибровочная диагностика;
- можно показать примеры high-confidence / low-confidence prediction.

## Команды проверки

```bash
python scripts/train_uncertainty.py
```

## Ограничения

- не усложнять до mixture density nets без причины;
- не ломать существующий regression pipeline.

## Что должен вернуть Codex после выполнения

1. модель uncertainty;
2. метрики mean + uncertainty;
3. краткий вывод, полезна ли неопределённость в проекте.
