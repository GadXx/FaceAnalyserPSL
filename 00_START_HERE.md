# Пошаговая реализация проекта для Codex

Этот набор файлов разбивает реализацию на **отдельные фазы**.  
Codex должен выполнять **только одну фазу за запуск**, не перескакивать через шаги и не пытаться реализовать весь проект сразу.

## Порядок выполнения

1. `01_PHASE_00_BOOTSTRAP.md`
2. `02_PHASE_01_DATASET_MANIFEST.md`
3. `03_PHASE_02_PREPROCESSING.md`
4. `04_PHASE_03_FEATURE_EXTRACTION.md`
5. `05_PHASE_04_DEEP_BASELINE.md`
6. `06_PHASE_05_GEOMETRY_BASELINE.md`
7. `07_PHASE_06_FUSION_MODEL.md`
8. `08_PHASE_07_RANKING_MODEL.md`
9. `09_PHASE_08_UNCERTAINTY_MODEL.md`
10. `10_PHASE_09_ROBUSTNESS_EVAL.md`
11. `11_PHASE_10_SUBGROUP_ANALYSIS.md`
12. `12_PHASE_11_INFERENCE_PIPELINE.md`
13. `13_PHASE_12_API.md`
14. `14_LOGGING_TESTING_REPORTS.md`

## Глобальная цель

Реализовать локальный исследовательский ML-проект для оценки facial aesthetics / face score на основе:
- готового face detector/alignment/embedding;
- собственного финального scorer-а;
- воспроизводимого data/feature/training pipeline.

## Рекомендуемый стек

- Python 3.11
- PyTorch
- InsightFace
- onnxruntime
- scikit-learn
- pandas / pyarrow
- XGBoost или LightGBM
- FastAPI
- Hydra или pydantic-settings

## Базовые правила для Codex

1. Реализовывать **только текущую фазу**.
2. Не переписывать уже готовые фазы без необходимости.
3. Все пути и параметры — через конфиг.
4. Не хардкодить абсолютные локальные пути.
5. После каждой фазы оставить:
   - код;
   - артефакт на диске;
   - короткий отчёт, что сделано;
   - список команд проверки.
6. Не обучать detector / face encoder с нуля.
7. Низкоуровневый CV использовать готовый; своё обучение — только на уровне task-specific head.

## Минимальная целевая структура репозитория

```text
face-aesthetics/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ features/
├─ scripts/
├─ src/face_aesthetics/
├─ reports/
└─ tests/
```

## Что считать успешной реализацией всего проекта

- есть CLI preprocessing pipeline;
- есть feature store;
- есть baseline-модели;
- есть fusion-модель;
- есть evaluation-отчёты;
- есть inference CLI;
- есть API;
- проект можно воспроизвести по README.
