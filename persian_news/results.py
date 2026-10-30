"""A common result record for trained models and helpers to compare them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ModelResult:
    name: str
    family: str  # "ml" or "dl"
    estimator: Any  # exposes .predict(list[list[str]]) -> encoded labels
    params: dict[str, Any]
    val_metrics: dict[str, float]
    train_seconds: float


def results_to_frame(results: list[ModelResult], sort_by: str) -> pd.DataFrame:
    rows = [
        {
            "model": result.name,
            "family": result.family,
            **{f"val_{name}": value for name, value in result.val_metrics.items()},
            "train_seconds": round(result.train_seconds, 1),
            "params": result.params,
        }
        for result in results
    ]
    return pd.DataFrame(rows).sort_values(f"val_{sort_by}", ascending=False).reset_index(drop=True)


def select_best(results: list[ModelResult], metric: str) -> ModelResult:
    return max(results, key=lambda result: result.val_metrics[metric])
