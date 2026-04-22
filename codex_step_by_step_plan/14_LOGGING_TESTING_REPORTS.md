# Логирование, тесты и отчёты

Этот файл не отдельная фаза обучения, а обязательный чек-лист качества для всех фаз.

## Что должно быть в проекте

### 1. Логирование
Добавить единый logging layer для:
- dataset preparation;
- preprocessing;
- feature extraction;
- training;
- evaluation;
- inference.

Минимально логировать:
- timestamp;
- phase;
- input path;
- output artifact path;
- counts processed/skipped/failed;
- key metrics.

### 2. Тесты
Минимальный набор тестов:
- manifest builder не создаёт дубликаты;
- preprocessing pipeline корректно обрабатывает "нет лица";
- feature extractor возвращает ожидаемую размерность;
- training script читает признаки и выдаёт метрики;
- inference CLI не падает на валидном изображении.

### 3. Отчёты
В `reports/` должны появляться:
- dataset summary;
- preprocessing summary;
- baseline metrics;
- fusion comparison;
- robustness report;
- subgroup analysis report;
- inference examples.

### 4. Финальный README
В README должны быть:
- структура проекта;
- команды установки;
- порядок фаз;
- команды запуска;
- где искать артефакты;
- как выполнить inference.

## Чек-лист перед завершением проекта

- [ ] bootstrap завершён
- [ ] manifest готов
- [ ] preprocessing работает
- [ ] features сохранены
- [ ] baseline deep обучен
- [ ] geometry baseline обучен
- [ ] fusion model обучена
- [ ] ranking проверен
- [ ] uncertainty проверен
- [ ] robustness посчитан
- [ ] subgroup analysis выполнен
- [ ] inference CLI готов
- [ ] API готов
- [ ] README обновлён
- [ ] тесты запускаются

## Как использовать этот набор файлов с Codex

На каждом запуске Codex:
1. открыть только один файл фазы;
2. выполнить только его;
3. после выполнения обновить README и список артефактов;
4. перейти к следующему файлу только после успешной проверки.
