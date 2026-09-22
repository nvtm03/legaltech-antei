"""REST API пайплайна: декомпозиция, поиск оговорки, стресс-тест."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.pipeline import InputError, run
from schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CLAUSES_DB = ROOT / "data" / "clauses_db.json"
_MAX_BODY = 256_000

app = FastAPI(title="LegalTech AI", version="1.0.0")


@app.middleware("http")
async def reject_large_body(request, call_next):
    """Не отдаёт модели тело больше 256 КБ. Проверяется и заголовок, и уже прочитанные байты."""
    raw = request.headers.get("content-length")
    if raw is not None:
        try:
            size = int(raw)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Некорректный Content-Length"})
        if size > _MAX_BODY:
            return JSONResponse(status_code=413, content={"detail": "Слишком большой запрос"})
    body = await request.body()
    if len(body) > _MAX_BODY:
        return JSONResponse(status_code=413, content={"detail": "Слишком большой запрос"})
    return await call_next(request)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    """Прогоняет договор через три этапа и возвращает модули, оговорку и риск."""
    if not CLAUSES_DB.is_file():
        raise HTTPException(status_code=500, detail="База оговорок не найдена")
    try:
        return run(body.contract_text, body.scenario, db_path=str(CLAUSES_DB))
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except Exception as exc:
        logger.error("Анализ договора не удался (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Анализ договора не удался") from None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
