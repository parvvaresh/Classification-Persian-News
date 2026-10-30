"""Neural text classifiers operating on padded token-index batches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..config import DLConfig
from .embeddings import PAD_INDEX


def _embedding_layer(
    vocab_size: int, embedding_dim: int, pretrained: torch.Tensor | None, freeze: bool
) -> nn.Embedding:
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
    if pretrained is not None:
        embedding.weight.data.copy_(pretrained)
    embedding.weight.requires_grad = not freeze
    return embedding


class TextCNN(nn.Module):
    """Parallel n-gram convolutions (kernel sizes 3/4/5 by default) with max-over-time pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        freeze_embeddings: bool = True,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding = _embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # `lengths` is unused: max pooling makes the CNN insensitive to trailing padding.
        x = self.embedding(ids).transpose(1, 2)  # (batch, embedding_dim, seq_len)
        pooled = [F.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class LSTMClassifier(nn.Module):
    """LSTM over the real (unpadded) tokens, classifying from the final hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        freeze_embeddings: bool = True,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding = _embedding_layer(vocab_size, embedding_dim, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Packing makes the LSTM stop at each title's last real token instead of reading padding.
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.lstm.bidirectional else h_n[-1]
        return self.fc(self.dropout(hidden))


MODEL_CLASSES: dict[str, type[nn.Module]] = {"cnn": TextCNN, "lstm": LSTMClassifier}


def model_kwargs(name: str, cfg: DLConfig, vocab_size: int, num_classes: int) -> dict[str, Any]:
    """Constructor arguments for ``name``; stored in the checkpoint to rebuild the model later."""
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": cfg.embedding_dim,
        "num_classes": num_classes,
        "freeze_embeddings": cfg.freeze_embeddings,
    }
    if name == "cnn":
        return {
            **common,
            "num_filters": cfg.cnn_num_filters,
            "kernel_sizes": tuple(cfg.cnn_kernel_sizes),
            "dropout": cfg.cnn_dropout,
        }
    if name == "lstm":
        return {
            **common,
            "hidden_size": cfg.lstm_hidden_size,
            "num_layers": cfg.lstm_num_layers,
            "bidirectional": cfg.lstm_bidirectional,
            "dropout": cfg.lstm_dropout,
        }
    raise ValueError(f"Unknown DL model {name!r}. Available: {sorted(MODEL_CLASSES)}")


def build_model(
    name: str, kwargs: dict[str, Any], pretrained_embeddings: torch.Tensor | None = None
) -> nn.Module:
    return MODEL_CLASSES[name](**kwargs, pretrained_embeddings=pretrained_embeddings)
