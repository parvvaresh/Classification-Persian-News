"""Exploratory data analysis: label distribution and a word cloud of preprocessed titles.

Figures are written to ``<output-dir>/eda/`` instead of being shown interactively.

Example:
    python eda.py --data data/farsnews_fainal.csv --font data/Gandom.ttf
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from persian_news.config import Config
from persian_news.data import LABEL_COLUMN, TOKENS_COLUMN, load_preprocessed_dataset
from persian_news.preprocessing import PersianTextPreprocessor, load_stopwords
from persian_news.utils import setup_logging

logger = logging.getLogger("eda")


def plot_label_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df[LABEL_COLUMN].value_counts()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color="#2E86C1", edgecolor="black", linewidth=1.5)
    ax.bar_label(bars, fontsize=12, fontweight="bold")
    ax.set_xlabel("Category", fontsize=14, labelpad=10)
    ax.set_ylabel("Count", fontsize=14, labelpad=10)
    ax.set_title("Data Distribution by Category", fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_word_cloud(df: pd.DataFrame, font_path: Path, path: Path, max_words: int = 200) -> None:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    from wordcloud import WordCloud

    frequencies = Counter(token for tokens in df[TOKENS_COLUMN] for token in tokens)
    # WordCloud draws left-to-right, so Persian words must be reshaped and reordered first.
    shaped = {get_display(reshape(word)): count for word, count in frequencies.most_common(max_words)}
    cloud = WordCloud(
        background_color="white",
        font_path=str(font_path),
        width=800,
        height=400,
        max_words=max_words,
        max_font_size=70,
        scale=5,
    ).generate_from_frequencies(shaped)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Processed Titles", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=defaults.paths.data_csv)
    parser.add_argument("--stopwords-dir", type=Path, default=defaults.paths.stopwords_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.paths.output_dir)
    parser.add_argument("--font", type=Path, help="Persian TTF font for the word cloud (skipped if omitted)")
    args = parser.parse_args()

    setup_logging()
    defaults.paths.output_dir = args.output_dir
    df = load_preprocessed_dataset(
        args.data,
        PersianTextPreprocessor(load_stopwords(args.stopwords_dir)),
        cache_file=defaults.paths.cache_file,
    )
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Label distribution:\n%s", df[LABEL_COLUMN].value_counts().to_string())

    figures_dir = args.output_dir / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_label_distribution(df, figures_dir / "label_distribution.png")
    if args.font:
        plot_word_cloud(df, args.font, figures_dir / "wordcloud.png")
    else:
        logger.info("No --font given; skipping the word cloud.")
    logger.info("Figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
