"""Train every ML and DL model on the Persian news dataset and save the best one.

All models are compared on the same held-out validation split using one selection
metric; the winner is then scored once on the untouched test split and saved.

Example:
    python train.py --data data/farsnews_fainal.csv --stopwords-dir data/persian_stopwords
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from persian_news.artifacts import save_best_model
from persian_news.config import Config
from persian_news.data import LABEL_COLUMN, TOKENS_COLUMN, load_preprocessed_dataset, split_dataset
from persian_news.dl.models import MODEL_CLASSES
from persian_news.dl.trainer import train_dl_models
from persian_news.metrics import METRIC_NAMES, compute_metrics
from persian_news.ml import train_ml_models
from persian_news.preprocessing import PersianTextPreprocessor, load_stopwords
from persian_news.results import ModelResult, results_to_frame, select_best
from persian_news.utils import get_device, set_seed, setup_logging

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    paths = parser.add_argument_group("paths")
    paths.add_argument("--data", type=Path, default=defaults.paths.data_csv, help="news CSV with 'title' and 'label' columns")
    paths.add_argument("--stopwords-dir", type=Path, default=defaults.paths.stopwords_dir, help="folder of *.txt stopword files")
    paths.add_argument("--output-dir", type=Path, default=defaults.paths.output_dir)
    paths.add_argument("--refresh-cache", action="store_true", help="re-run preprocessing even if a cache exists")

    general = parser.add_argument_group("general")
    general.add_argument("--metric", choices=METRIC_NAMES, default=defaults.selection_metric, help="model selection metric")
    general.add_argument("--seed", type=int, default=defaults.seed)
    general.add_argument("--test-size", type=float, default=defaults.split.test_size)
    general.add_argument("--val-size", type=float, default=defaults.split.val_size)

    ml = parser.add_argument_group("machine learning")
    ml.add_argument("--skip-ml", action="store_true")
    ml.add_argument("--ml-models", nargs="+", help="subset of ML models to train (default: all)")
    ml.add_argument("--tfidf-max-features", type=int, default=defaults.ml.tfidf_max_features, help="0 = full vocabulary")
    ml.add_argument("--tfidf-max-ngram", type=int, default=defaults.ml.tfidf_ngram_range[1])
    ml.add_argument("--cv-folds", type=int, default=defaults.ml.cv_folds)
    ml.add_argument("--n-jobs", type=int, default=defaults.ml.n_jobs)

    dl = parser.add_argument_group("deep learning")
    dl.add_argument("--skip-dl", action="store_true")
    dl.add_argument("--dl-models", nargs="+", choices=sorted(MODEL_CLASSES), default=defaults.dl.models)
    dl.add_argument("--epochs", type=int, default=defaults.dl.num_epochs)
    dl.add_argument("--batch-size", type=int, default=defaults.dl.batch_size)
    dl.add_argument("--lr", type=float, default=defaults.dl.learning_rate)
    dl.add_argument("--patience", type=int, default=defaults.dl.patience)
    dl.add_argument("--train-embeddings", action="store_true", help="fine-tune Word2Vec embeddings instead of freezing them")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(selection_metric=args.metric, seed=args.seed, refresh_cache=args.refresh_cache)

    cfg.paths.data_csv = args.data
    cfg.paths.stopwords_dir = args.stopwords_dir
    cfg.paths.output_dir = args.output_dir

    cfg.split.test_size = args.test_size
    cfg.split.val_size = args.val_size

    cfg.ml.enabled = not args.skip_ml
    cfg.ml.models = args.ml_models
    cfg.ml.tfidf_max_features = args.tfidf_max_features or None
    cfg.ml.tfidf_ngram_range = (1, args.tfidf_max_ngram)
    cfg.ml.cv_folds = args.cv_folds
    cfg.ml.n_jobs = args.n_jobs

    cfg.dl.enabled = not args.skip_dl
    cfg.dl.models = args.dl_models
    cfg.dl.num_epochs = args.epochs
    cfg.dl.batch_size = args.batch_size
    cfg.dl.learning_rate = args.lr
    cfg.dl.patience = args.patience
    cfg.dl.freeze_embeddings = not args.train_embeddings
    return cfg


def main() -> None:
    setup_logging()
    cfg = build_config(parse_args())
    set_seed(cfg.seed)

    stopwords = load_stopwords(cfg.paths.stopwords_dir)
    df = load_preprocessed_dataset(
        cfg.paths.data_csv,
        PersianTextPreprocessor(stopwords),
        cache_file=cfg.paths.cache_file,
        refresh_cache=cfg.refresh_cache,
    )
    logger.info("Label distribution:\n%s", df[LABEL_COLUMN].value_counts().to_string())
    splits = split_dataset(df, cfg.split, cfg.seed)
    label_encoder = LabelEncoder().fit(df[LABEL_COLUMN])

    results: list[ModelResult] = []
    if cfg.ml.enabled:
        results += train_ml_models(splits, label_encoder, cfg.ml, cfg.selection_metric, cfg.seed)
    if cfg.dl.enabled:
        results += train_dl_models(splits, label_encoder, cfg.dl, cfg.selection_metric, cfg.seed, get_device())
    if not results:
        raise SystemExit("No model was trained (both ML and DL disabled, or every model failed).")

    leaderboard = results_to_frame(results, cfg.selection_metric)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(cfg.paths.results_file, index=False)
    logger.info("Validation leaderboard:\n%s", leaderboard.drop(columns="params").to_string(index=False))
    logger.info("Leaderboard saved to %s", cfg.paths.results_file)

    best = select_best(results, cfg.selection_metric)
    y_test = label_encoder.transform(splits.test[LABEL_COLUMN])
    test_metrics = compute_metrics(y_test, best.estimator.predict(splits.test[TOKENS_COLUMN].tolist()))
    logger.info(
        "Best model: %s (%s) | val %s=%.4f | test metrics: %s",
        best.name,
        best.family,
        cfg.selection_metric,
        best.val_metrics[cfg.selection_metric],
        {name: round(value, 4) for name, value in test_metrics.items()},
    )

    save_best_model(best, label_encoder, stopwords, cfg.selection_metric, test_metrics, cfg.paths.best_model_dir)


if __name__ == "__main__":
    main()
