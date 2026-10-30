"""Configuration objects for the Persian news classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    data_csv: Path = Path("data/farsnews_fainal.csv")
    stopwords_dir: Path | None = Path("data/persian_stopwords")
    output_dir: Path = Path("artifacts")

    @property
    def cache_file(self) -> Path:
        return self.output_dir / "cache" / "preprocessed.pkl"

    @property
    def results_file(self) -> Path:
        return self.output_dir / "results.csv"

    @property
    def best_model_dir(self) -> Path:
        return self.output_dir / "best_model"


@dataclass
class SplitConfig:
    # Both fractions are relative to the full dataset.
    test_size: float = 0.2
    val_size: float = 0.1


@dataclass
class MLConfig:
    enabled: bool = True
    models: list[str] | None = None  # None -> every model in the zoo
    tfidf_max_features: int | None = 100  # None -> full vocabulary
    tfidf_ngram_range: tuple[int, int] = (1, 1)
    cv_folds: int = 5
    n_jobs: int = -1


@dataclass
class DLConfig:
    enabled: bool = True
    models: list[str] = field(default_factory=lambda: ["cnn", "lstm"])

    # Word2Vec embeddings
    embedding_dim: int = 100
    w2v_window: int = 5
    w2v_min_count: int = 1
    w2v_epochs: int = 5
    w2v_workers: int = 4
    freeze_embeddings: bool = True
    max_seq_len: int | None = None  # None -> longest title in the training split

    # Optimisation
    batch_size: int = 100
    learning_rate: float = 1e-3
    num_epochs: int = 25
    patience: int = 5  # early stopping on the validation selection metric

    # TextCNN
    cnn_num_filters: int = 100
    cnn_kernel_sizes: tuple[int, ...] = (3, 4, 5)
    cnn_dropout: float = 0.5

    # LSTM
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_bidirectional: bool = False
    lstm_dropout: float = 0.0


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    dl: DLConfig = field(default_factory=DLConfig)
    selection_metric: str = "kappa"  # used for grid search, early stopping and picking the best model
    seed: int = 42
    refresh_cache: bool = False
