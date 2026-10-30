"""Classify Persian news titles with the best model saved by ``train.py``.

Examples:
    python predict.py "پرسپولیس با برتری مقابل استقلال قهرمان شد"
    python predict.py --file titles.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from persian_news.artifacts import NewsClassifier
from persian_news.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", help="news titles to classify")
    parser.add_argument("--file", type=Path, help="text file with one title per line")
    parser.add_argument("--model-dir", type=Path, default=Config().paths.best_model_dir)
    args = parser.parse_args()

    texts = list(args.texts)
    if args.file:
        texts += [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        parser.error("pass at least one title or --file")

    classifier = NewsClassifier(args.model_dir)
    for text, label in zip(texts, classifier.predict(texts)):
        print(f"{label}\t{text}")


if __name__ == "__main__":
    main()
