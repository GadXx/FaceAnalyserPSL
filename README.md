# FaceAnalyserPSL

Bootstrap-каркас для локального ML-пайплайна `facial aesthetics scoring`.

На текущем этапе в репозитории подготовлены:
- структура каталогов проекта;
- базовые YAML-конфиги;
- файл зависимостей;
- минимальный Python-пакет;
- smoke-check для проверки импортов и чтения конфигов.

## Структура

```text
configs/
data/
  raw/
  interim/
  processed/
  features/
reports/
scripts/
src/
  face_aesthetics/
tests/
```

## Установка

```bash
python -m pip install -r requirements.txt
```

## Smoke Check

```bash
python -m src.face_aesthetics.core.smoke_check
```

Smoke-check загружает конфиги из `configs/`, печатает ключевые пути и проверяет, что базовые импорты проекта работают.
