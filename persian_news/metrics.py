"""Evaluation metrics shared by every model family."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

METRIC_NAMES = ("accuracy", "precision", "recall", "f1_weighted", "f1_macro", "kappa")


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
