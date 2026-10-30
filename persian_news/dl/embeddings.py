"""Word2Vec embeddings and the token-to-index vocabulary built on top of them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from gensim.models import Word2Vec

from ..config import DLConfig

PAD_TOKEN = "<pad>"
PAD_INDEX = 0


class Vocabulary:
    """Maps tokens to embedding rows; row 0 is reserved for padding."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.itos = [PAD_TOKEN, *tokens]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def tokens(self) -> list[str]:
        return self.itos[1:]

    def encode(self, tokens: Sequence[str], max_len: int) -> list[int]:
        """Out-of-vocabulary tokens are dropped, then the sequence is truncated to ``max_len``."""
        return [self.stoi[token] for token in tokens if token in self.stoi][:max_len]


def encode_batch(
    token_lists: Sequence[Sequence[str]], vocab: Vocabulary, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded index matrix ``(n, max_len)`` and true lengths (at least 1, for LSTM packing)."""
    ids = torch.full((len(token_lists), max_len), PAD_INDEX, dtype=torch.long)
    lengths = torch.ones(len(token_lists), dtype=torch.long)
    for row, tokens in enumerate(token_lists):
        encoded = vocab.encode(tokens, max_len)
        if encoded:
            ids[row, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
            lengths[row] = len(encoded)
    return ids, lengths


def train_word2vec(sentences: list[list[str]], cfg: DLConfig, seed: int) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=cfg.embedding_dim,
        window=cfg.w2v_window,
        min_count=cfg.w2v_min_count,
        epochs=cfg.w2v_epochs,
        workers=cfg.w2v_workers,
        seed=seed,
    )


def build_vocabulary(w2v: Word2Vec) -> tuple[Vocabulary, torch.Tensor]:
    """Vocabulary plus an embedding matrix whose row ``i`` is the vector of ``vocab.itos[i]``."""
    vocab = Vocabulary(w2v.wv.index_to_key)
    weights = np.zeros((len(vocab), w2v.wv.vector_size), dtype=np.float32)
    weights[1:] = w2v.wv.vectors
    return vocab, torch.from_numpy(weights)
