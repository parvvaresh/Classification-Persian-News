"""Classic machine-learning models: TF-IDF features + grid-searched classifiers."""

from __future__ import annotations

import logging
import time
from typing import Any

from sklearn import ensemble, linear_model, naive_bayes, neighbors, tree
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import MLConfig
from .data import LABEL_COLUMN, TOKENS_COLUMN, DataSplits
from .metrics import compute_metrics
from .results import ModelResult

logger = logging.getLogger(__name__)

ParamGrid = dict[str, list[Any]]

_SCORERS = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
    "kappa": make_scorer(cohen_kappa_score),
}


def _identity(tokens):
    """Input is already tokenized by PersianTextPreprocessor; must be module-level to be picklable."""
    return tokens


def build_model_zoo(seed: int) -> dict[str, tuple[BaseEstimator, ParamGrid]]:
    zoo: dict[str, tuple[BaseEstimator, ParamGrid]] = {
        # Ensemble methods
        "AdaBoostClassifier": (
            ensemble.AdaBoostClassifier(random_state=seed),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
        ),
        "BaggingClassifier": (
            ensemble.BaggingClassifier(random_state=seed),
            {"n_estimators": [10, 50, 100], "max_samples": [0.5, 1.0], "max_features": [0.5, 1.0]},
        ),
        "RandomForestClassifier": (
            ensemble.RandomForestClassifier(random_state=seed),
            {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        "ExtraTreesClassifier": (
            ensemble.ExtraTreesClassifier(random_state=seed),
            {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        # Linear models
        "PassiveAggressiveClassifier": (
            linear_model.PassiveAggressiveClassifier(random_state=seed),
            {"C": [0.1, 1.0, 10.0], "max_iter": [1000, 2000]},
        ),
        "SGDClassifier": (
            linear_model.SGDClassifier(random_state=seed),
            {
                "loss": ["hinge", "log_loss"],
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01],
                "max_iter": [1000, 2000],
            },
        ),
        "Perceptron": (
            linear_model.Perceptron(random_state=seed),
            {"penalty": [None, "l2", "l1", "elasticnet"], "alpha": [0.0001, 0.001, 0.01], "max_iter": [1000, 2000]},
        ),
        "LogisticRegression": (
            linear_model.LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        # Naive Bayes
        "BernoulliNB": (
            naive_bayes.BernoulliNB(),
            {"alpha": [0.1, 0.5, 1.0], "binarize": [0.0, 0.5, 1.0]},
        ),
        "MultinomialNB": (
            naive_bayes.MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0]},
        ),
        # Nearest neighbours
        "KNeighborsClassifier": (
            neighbors.KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
        ),
        # Trees
        "DecisionTreeClassifier": (
            tree.DecisionTreeClassifier(random_state=seed),
            {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
        "ExtraTreeClassifier": (
            tree.ExtraTreeClassifier(random_state=seed),
            {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        ),
    }

    try:
        from xgboost import XGBClassifier
    except Exception as error:  # not just ImportError: macOS without libomp raises XGBoostError
        logger.warning("xgboost is unavailable (%s); XGBClassifier will be skipped.", error)
    else:
        # n_jobs=1: GridSearchCV already parallelises across candidates.
        zoo["XGBClassifier"] = (
            XGBClassifier(eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=seed, verbosity=0),
            {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
        )
    return zoo


def build_pipeline(classifier: BaseEstimator, cfg: MLConfig) -> Pipeline:
    # TF-IDF lives inside the pipeline so it is fitted on training folds only (no leakage).
    vectorizer = TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        lowercase=False,
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_ml_models(
    splits: DataSplits,
    label_encoder: LabelEncoder,
    cfg: MLConfig,
    selection_metric: str,
    seed: int,
) -> list[ModelResult]:
    zoo = build_model_zoo(seed)
    names = cfg.models or list(zoo)
    unknown = sorted(set(names) - set(zoo))
    if unknown:
        raise ValueError(f"Unknown ML models {unknown}. Available: {sorted(zoo)}")

    X_train = splits.train[TOKENS_COLUMN].tolist()
    X_val = splits.val[TOKENS_COLUMN].tolist()
    y_train = label_encoder.transform(splits.train[LABEL_COLUMN])
    y_val = label_encoder.transform(splits.val[LABEL_COLUMN])
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)

    results: list[ModelResult] = []
    for name in names:
        classifier, grid = zoo[name]
        search = GridSearchCV(
            build_pipeline(classifier, cfg),
            param_grid={f"clf__{key}": values for key, values in grid.items()},
            scoring=_SCORERS[selection_metric],
            cv=cv,
            n_jobs=cfg.n_jobs,
            refit=True,
        )

        logger.info("[ML] Grid-searching %s ...", name)
        start = time.perf_counter()
        try:
            search.fit(X_train, y_train)
        except Exception:
            logger.exception("[ML] %s failed; skipping it.", name)
            continue
        elapsed = time.perf_counter() - start

        val_metrics = compute_metrics(y_val, search.predict(X_val))
        params = {key.removeprefix("clf__"): value for key, value in search.best_params_.items()}
        logger.info(
            "[ML] %s | best params %s | val %s=%.4f | %.1fs",
            name, params, selection_metric, val_metrics[selection_metric], elapsed,
        )
        results.append(
            ModelResult(
                name=name,
                family="ml",
                estimator=search.best_estimator_,
                params={
                    **params,
                    "tfidf_max_features": cfg.tfidf_max_features,
                    "tfidf_ngram_range": cfg.tfidf_ngram_range,
                },
                val_metrics=val_metrics,
                train_seconds=elapsed,
            )
        )
    return results
