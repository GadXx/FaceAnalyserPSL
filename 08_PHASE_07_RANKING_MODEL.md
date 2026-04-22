# Фаза 7 — Pairwise ranking model

## Цель

Проверить, даёт ли ranking-постановка преимущество над прямой регрессией на субъективной шкале.

## Зависимости

Требуются завершённые:
- `Фаза 6`

## Что должен сделать Codex

1. Реализовать `scripts/train_ranking.py`.
2. Сгенерировать пары:
   - `(image_a, image_b, label)`
   - где `label = 1`, если `score_a > score_b`

3. Использовать признаки:
   - deep-only **или**
   - fusion-вектор

4. Реализовать один из вариантов:
   - pairwise logistic loss
   - margin ranking loss
   - Bradley-Terry style scoring

5. Метрики:
   - pairwise accuracy
   - Spearman после восстановления scalar score, если это поддерживается архитектурой

6. Сравнить ranking с лучшей regression-моделью.

## Критерии завершения

- ranking pipeline работает;
- есть pair generation;
- есть сравнение с regression.

## Команды проверки

```bash
python scripts/train_ranking.py
```

## Ограничения

- не усложнять Siamese-архитектуру без необходимости;
- не тянуть тяжёлые backbone-обучения.

## Что должен вернуть Codex после выполнения

1. ranking metric;
2. сравнение с regression;
3. решение: ranking оставлять как дополнительную ветку или нет.
