"""Loading, preprocessing (with caching) and splitting of the news dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .config import SplitConfig
from .preprocessing import PersianTextPreprocessor

logger = logging.getLogger(__name__)

TEXT_COLUMN = "title"
LABEL_COLUMN = "label"
TOKENS_COLUMN = "tokens"


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=[TEXT_COLUMN, LABEL_COLUMN])
    n_rows = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    logger.info(
        "Loaded %d rows from %s (%d empty/duplicate rows dropped).", len(df), csv_path, n_rows - len(df)
    )
    return df


def load_preprocessed_dataset(
    csv_path: Path,
    preprocessor: PersianTextPreprocessor,
    cache_file: Path | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Return the dataset with a ``tokens`` column, reusing the on-disk cache when possible."""
    if cache_file is not None and cache_file.exists() and not refresh_cache:
        logger.info("Loading preprocessed dataset from cache %s", cache_file)
        return pd.read_pickle(cache_file)

    df = load_raw_dataset(csv_path)
    tqdm.pandas(desc="Preprocessing")
    df[TOKENS_COLUMN] = df[TEXT_COLUMN].progress_apply(preprocessor)

    empty = df[TOKENS_COLUMN].map(len) == 0
    if empty.any():
        logger.info("Dropping %d rows with no tokens left after preprocessing.", int(empty.sum()))
    df = df[~empty].reset_index(drop=True)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_file)
        logger.info("Cached preprocessed dataset to %s", cache_file)
    return df


def split_dataset(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> DataSplits:
    """Stratified train / validation / test split."""
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[LABEL_COLUMN], random_state=seed
    )
    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[LABEL_COLUMN], random_state=seed
    )
    logger.info("Split sizes: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
