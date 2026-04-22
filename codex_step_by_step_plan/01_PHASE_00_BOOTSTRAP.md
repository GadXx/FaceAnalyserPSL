# Фаза 0 — Bootstrap проекта

## Цель

Подготовить репозиторий, зависимости, конфиги и каркас проекта так, чтобы следующие фазы можно было выполнять без хаоса.

## Зависимости от предыдущих шагов

Нет. Это первая фаза.

## Что должен сделать Codex

1. Создать структуру каталогов:
   - `configs/`
   - `data/raw`, `data/interim`, `data/processed`, `data/features`
   - `scripts/`
   - `src/face_aesthetics/`
   - `reports/`
   - `tests/`

2. Добавить:
   - `README.md`
   - `pyproject.toml` **или** `requirements.txt`
   - `.gitignore`
   - `.env.example`

3. Настроить базовые зависимости:
   - numpy
   - pandas
   - pyarrow
   - scikit-learn
   - torch
   - torchvision
   - opencv-python
   - insightface
   - onnxruntime
   - xgboost
   - lightgbm
   - matplotlib
   - fastapi
   - uvicorn
   - pydantic
   - pyyaml
   - hydra-core

4. Создать конфиги:
   - `configs/base.yaml`
   - `configs/data.yaml`
   - `configs/preprocess.yaml`
   - `configs/train_baseline.yaml`
   - `configs/train_fusion.yaml`
   - `configs/infer.yaml`

5. Создать пустые каркасы модулей:
   - `src/face_aesthetics/core/`
   - `src/face_aesthetics/data/`
   - `src/face_aesthetics/features/`
   - `src/face_aesthetics/models/`
   - `src/face_aesthetics/pipelines/`
   - `src/face_aesthetics/services/`
   - `src/face_aesthetics/utils/`

6. Добавить простой smoke script, который:
   - читает конфиг;
   - печатает пути;
   - не падает на import.

## Артефакты после фазы

- рабочая структура проекта;
- файл зависимостей;
- конфиги;
- базовый README;
- smoke script.

## Критерии завершения

- проект устанавливается;
- импорты работают;
- есть единый конфиг;
- можно выполнить smoke command без ошибок.

## Команды проверки

```bash
python -m pip install -r requirements.txt
python -m src.face_aesthetics.core.smoke_check
```

## Ограничения

- не реализовывать ещё preprocessing;
- не писать training logic;
- не тащить лишние зависимости;
- не использовать notebook как основной способ запуска.

## Что должен вернуть Codex после выполнения

1. список созданных файлов;
2. команды запуска;
3. подтверждение, что структура готова к следующей фазе.
