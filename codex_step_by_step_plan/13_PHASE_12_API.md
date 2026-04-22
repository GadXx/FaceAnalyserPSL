# Фаза 12 — API

## Цель

Обернуть inference pipeline в локальный API.

## Зависимости

Требуется завершённая:
- `Фаза 11`

## Что должен сделать Codex

1. Реализовать `FastAPI` сервис.
2. Добавить endpoint:
   - `POST /score`

3. На вход принимать:
   - изображение;
   - optional flags для debug mode.

4. На выход возвращать:
   - `score`
   - `uncertainty` если поддерживается
   - `quality_pass`
   - `failure_reason` если инференс отклонён
   - технические метаданные в debug mode

5. Добавить:
   - startup loading model;
   - health endpoint;
   - простую обработку исключений.

## Критерии завершения

- API запускается локально;
- `/health` отвечает;
- `/score` работает на одном изображении.

## Команды проверки

```bash
uvicorn src.face_aesthetics.api.app:app --reload
```

## Ограничения

- не добавлять auth, БД и очереди;
- это локальный исследовательский API, а не production backend.

## Что должен вернуть Codex после выполнения

1. список endpoint-ов;
2. пример запроса;
3. пример ответа.
