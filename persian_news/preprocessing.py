"""Persian text cleaning, tokenization and lemmatization."""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Iterable
from pathlib import Path

import hazm

logger = logging.getLogger(__name__)

PERSIAN_PUNCTUATIONS = "`÷×؛#<>_()*&^%][ـ،/:\"؟.,'{}~¦+|!”…“–"
PERSIAN_LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

_PUNCTUATION_TABLE = str.maketrans({ch: " " for ch in string.punctuation + PERSIAN_PUNCTUATIONS})
_ARABIC_TO_PERSIAN_TABLE = str.maketrans(
    {"إ": "ا", "أ": "ا", "آ": "ا", "ي": "ی", "ئ": "ی", "ؤ": "و", "ة": "ه", "ك": "ک"}
)
# Tanwin, fatha, damma, kasra, shadda, sukun and tatweel (kashida).
_DIACRITICS_RE = re.compile("[ً-ْـ]")
# Collapse 3+ repeats ("سلاااام" -> "سلام") but keep legitimate double letters.
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_NON_PERSIAN_RE = re.compile(f"[^{PERSIAN_LETTERS}]")
_WHITESPACE_RE = re.compile(r"\s+")


class PersianTextPreprocessor:
    """Turns a raw Persian string into a list of cleaned, lemmatized tokens."""

    def __init__(self, stopwords: Iterable[str] = (), min_token_length: int = 3) -> None:
        self.stopwords = frozenset(stopwords)
        self.min_token_length = min_token_length
        self._normalizer = hazm.Normalizer()
        self._lemmatizer = hazm.Lemmatizer()

    def __call__(self, text: object) -> list[str]:
        if not isinstance(text, str):
            return []
        tokens = self._filter(hazm.word_tokenize(self.clean(text)))
        return self._filter(self._lemmatizer.lemmatize(token) for token in tokens)

    def clean(self, text: str) -> str:
        text = _DIACRITICS_RE.sub("", text)
        text = text.translate(_PUNCTUATION_TABLE)
        text = _REPEATED_CHAR_RE.sub(r"\1", text)
        text = text.translate(_ARABIC_TO_PERSIAN_TABLE)
        text = _NON_PERSIAN_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return self._normalizer.normalize(text)

    def _filter(self, tokens: Iterable[str]) -> list[str]:
        return [
            token
            for token in tokens
            if len(token) >= self.min_token_length and token not in self.stopwords
        ]


def load_stopwords(directory: Path | None = None) -> set[str]:
    """hazm's stopword list, extended with every ``*.txt`` file (one word per line) in ``directory``."""
    stopwords = set(hazm.stopwords_list())
    if directory is None:
        return stopwords

    files = sorted(Path(directory).glob("*.txt"))
    if not files:
        logger.warning("No stopword files found in %s; using hazm's list only.", directory)
    for path in files:
        with path.open(encoding="utf-8") as f:
            stopwords.update(line.strip() for line in f if line.strip())
    logger.info("Loaded %d stopwords.", len(stopwords))
    return stopwords
