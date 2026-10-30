"""Saving the selected best model and loading it back for inference.

Layout of the saved directory::

    best_model/
        metadata.json   # model name, family, labels, params, val/test metrics
        stopwords.txt   # stopwords used by the preprocessor at training time
        model.joblib    # ML: full TF-IDF + classifier sklearn pipeline
        model.pt        # DL: state_dict + constructor kwargs + vocabulary
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder

from .dl.trainer import DLPredictor
from .preprocessing import PersianTextPreprocessor
from .results import ModelResult
from .utils import get_device

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
STOPWORDS_FILE = "stopwords.txt"
ML_MODEL_FILE = "model.joblib"
DL_MODEL_FILE = "model.pt"


def save_best_model(
    result: ModelResult,
    label_encoder: LabelEncoder,
    stopwords: Iterable[str],
    selection_metric: str,
    test_metrics: dict[str, float],
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if result.family == "ml":
        joblib.dump(result.estimator, output_dir / ML_MODEL_FILE)
    else:
        torch.save(result.estimator.checkpoint(), output_dir / DL_MODEL_FILE)

    (output_dir / STOPWORDS_FILE).write_text("\n".join(sorted(stopwords)), encoding="utf-8")

    metadata = {
        "name": result.name,
        "family": result.family,
        "labels": label_encoder.classes_.tolist(),
        "selection_metric": selection_metric,
        "params": result.params,
        "val_metrics": result.val_metrics,
        "test_metrics": test_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved best model (%s) to %s", result.name, output_dir)
    return output_dir


class NewsClassifier:
    """Loads a saved best model and classifies raw Persian news titles."""

    def __init__(self, model_dir: Path, device: torch.device | None = None) -> None:
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
        self.labels = np.asarray(self.metadata["labels"])

        stopwords = (model_dir / STOPWORDS_FILE).read_text(encoding="utf-8").splitlines()
        self.preprocessor = PersianTextPreprocessor(stopwords)

        if self.metadata["family"] == "ml":
            self.estimator = joblib.load(model_dir / ML_MODEL_FILE)
        else:
            checkpoint = torch.load(model_dir / DL_MODEL_FILE, map_location="cpu", weights_only=True)
            self.estimator = DLPredictor.from_checkpoint(checkpoint, device or get_device())

    def predict(self, texts: Sequence[str]) -> list[str]:
        token_lists = [self.preprocessor(text) for text in texts]
        return self.labels[self.estimator.predict(token_lists)].tolist()

    @property
    def has_probabilities(self) -> bool:
        """False when scores are a softmax over margins (e.g. Perceptron), not real probabilities."""
        return self.metadata["family"] == "dl" or hasattr(self.estimator, "predict_proba")

    def score(self, text: str) -> dict[str, float] | None:
        """Score per label, highest first; ``None`` if no token survives preprocessing."""
        tokens = self.preprocessor(text)
        if not tokens:
            return None

        if self.metadata["family"] == "dl":
            scores = softmax(self.estimator.predict_logits([tokens]).numpy(), axis=1)
        elif self.has_probabilities:
            scores = self.estimator.predict_proba([tokens])
        else:
            scores = softmax(self.estimator.decision_function([tokens]), axis=1)

        ranked = sorted(zip(self.labels.tolist(), scores[0].tolist()), key=lambda pair: pair[1], reverse=True)
        return dict(ranked)
