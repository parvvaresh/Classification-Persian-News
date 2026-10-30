"""HTTP API and web page for the best model saved by ``train.py``.

The server also starts without a trained model: the page then explains how to train one.
Restart it after training to load the new model.

Examples:
    python api.py
    python api.py --model-dir artifacts/best_model --port 8080
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from persian_news.artifacts import METADATA_FILE, NewsClassifier
from persian_news.config import Config
from persian_news.utils import setup_logging

logger = logging.getLogger("api")

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"
MAX_TEXT_LENGTH = 5000


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class LabelScore(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    score: float
    is_probability: bool
    scores: list[LabelScore]


class ModelInfo(BaseModel):
    ready: bool
    name: str | None = None
    family: str | None = None
    labels: list[str] = []
    test_metrics: dict[str, float] = {}
    created_at: str | None = None


def load_classifier(model_dir: Path) -> NewsClassifier | None:
    if not (model_dir / METADATA_FILE).exists():
        logger.warning("No trained model in %s; /api/predict will answer 503. Run train.py first.", model_dir)
        return None
    # CPU is plenty for one title at a time.
    classifier = NewsClassifier(model_dir, device=torch.device("cpu"))
    logger.info("Loaded %s from %s", classifier.metadata["name"], model_dir)
    return classifier


def create_app(model_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.classifier = load_classifier(model_dir)
        yield

    app = FastAPI(title="Persian News Classifier", lifespan=lifespan)

    @app.get("/api/model")
    def model_info(request: Request) -> ModelInfo:
        classifier: NewsClassifier | None = request.app.state.classifier
        if classifier is None:
            return ModelInfo(ready=False)
        metadata: dict[str, Any] = classifier.metadata
        return ModelInfo(
            ready=True,
            name=metadata["name"],
            family=metadata["family"],
            labels=metadata["labels"],
            test_metrics=metadata["test_metrics"],
            created_at=metadata["created_at"],
        )

    # A plain `def`: FastAPI runs it in a worker thread, so inference does not block the event loop.
    @app.post("/api/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        classifier: NewsClassifier | None = request.app.state.classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="مدل هنوز آموزش داده نشده است.")

        scores = classifier.score(body.text)
        if scores is None:
            raise HTTPException(status_code=422, detail="در این متن واژهٔ فارسی قابل‌تحلیلی پیدا نشد.")

        label, score = next(iter(scores.items()))
        return PredictResponse(
            label=label,
            score=score,
            is_probability=classifier.has_probabilities,
            scores=[LabelScore(label=name, score=value) for name, value in scores.items()],
        )

    # Mounted last: "/" would otherwise shadow the API routes above.
    if (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    else:
        logger.warning("No frontend build in %s; run `npm install && npm run build` in frontend/.", FRONTEND_DIR)

        @app.get("/", include_in_schema=False)
        def index() -> PlainTextResponse:
            return PlainTextResponse("Frontend not built. Run `npm install && npm run build` in frontend/, then restart.")

    return app


app = create_app(Config().paths.best_model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=Config().paths.best_model_dir)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    setup_logging()
    uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
