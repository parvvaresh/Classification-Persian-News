"""Training loop with early stopping, and an inference wrapper for the neural models."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ..config import DLConfig
from ..data import LABEL_COLUMN, TOKENS_COLUMN, DataSplits
from ..metrics import compute_metrics
from ..results import ModelResult
from ..utils import set_seed
from .embeddings import Vocabulary, build_vocabulary, encode_batch, train_word2vec
from .models import MODEL_CLASSES, build_model, model_kwargs

logger = logging.getLogger(__name__)


class DLPredictor:
    """Bundles a trained network with its vocabulary so it can predict straight from tokens."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        kwargs: dict[str, Any],
        vocab: Vocabulary,
        max_len: int,
        device: torch.device,
        batch_size: int = 256,
    ) -> None:
        self.name = name
        self.model = model.to(device)
        self.kwargs = kwargs
        self.vocab = vocab
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_logits(self, token_lists: Sequence[Sequence[str]]) -> torch.Tensor:
        num_classes = self.kwargs["num_classes"]
        if not token_lists:
            return torch.empty(0, num_classes)
        ids, lengths = encode_batch(token_lists, self.vocab, self.max_len)
        loader = DataLoader(TensorDataset(ids, lengths), batch_size=self.batch_size)
        self.model.eval()
        return torch.cat(
            [self.model(batch_ids.to(self.device), batch_lengths).cpu() for batch_ids, batch_lengths in loader]
        )

    def predict(self, token_lists: Sequence[Sequence[str]]) -> np.ndarray:
        return self.predict_logits(token_lists).argmax(dim=1).numpy()

    def checkpoint(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_kwargs": self.kwargs,
            "state_dict": {key: value.cpu() for key, value in self.model.state_dict().items()},
            "vocab": self.vocab.tokens,
            "max_len": self.max_len,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any], device: torch.device) -> DLPredictor:
        model = build_model(checkpoint["name"], checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            name=checkpoint["name"],
            model=model,
            kwargs=checkpoint["model_kwargs"],
            vocab=Vocabulary(checkpoint["vocab"]),
            max_len=checkpoint["max_len"],
            device=device,
        )


def _make_loader(
    df: pd.DataFrame,
    vocab: Vocabulary,
    max_len: int,
    label_encoder: LabelEncoder,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ids, lengths = encode_batch(df[TOKENS_COLUMN].tolist(), vocab, max_len)
    labels = torch.as_tensor(label_encoder.transform(df[LABEL_COLUMN]), dtype=torch.long)
    return DataLoader(TensorDataset(ids, lengths, labels), batch_size=batch_size, shuffle=shuffle)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, lengths), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, predictions = [], []
    for ids, lengths, labels in loader:
        predictions.append(model(ids.to(device), lengths).argmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(predictions).numpy()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DLConfig,
    selection_metric: str,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    """Train with early stopping; restores the best epoch's weights and returns its val metrics."""
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_score, best_epoch = -math.inf, 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = compute_metrics(*_predict_loader(model, val_loader, device))
        score = val_metrics[selection_metric]
        logger.info(
            "  epoch %02d/%d | train loss %.4f | val %s %.4f",
            epoch, cfg.num_epochs, train_loss, selection_metric, score,
        )

        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, val_metrics
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            logger.info("  early stopping: no improvement for %d epochs", cfg.patience)
            break

    model.load_state_dict(best_state)
    return best_metrics, best_epoch


def train_dl_models(
    splits: DataSplits,
    label_encoder: LabelEncoder,
    cfg: DLConfig,
    selection_metric: str,
    seed: int,
    device: torch.device,
) -> list[ModelResult]:
    set_seed(seed)
    train_tokens = splits.train[TOKENS_COLUMN].tolist()

    # Embeddings are learned from the training split only, so val/test stay unseen.
    logger.info("[DL] Training Word2Vec on %d titles ...", len(train_tokens))
    vocab, embeddings = build_vocabulary(train_word2vec(train_tokens, cfg, seed))

    max_len = cfg.max_seq_len or max(len(tokens) for tokens in train_tokens)
    max_len = max(max_len, *cfg.cnn_kernel_sizes)  # every convolution needs at least `kernel_size` steps
    logger.info("[DL] vocab size %d | max sequence length %d | device %s", len(vocab), max_len, device)

    train_loader = _make_loader(splits.train, vocab, max_len, label_encoder, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(splits.val, vocab, max_len, label_encoder, cfg.batch_size, shuffle=False)
    num_classes = len(label_encoder.classes_)

    results: list[ModelResult] = []
    for name in cfg.models:
        set_seed(seed)
        kwargs = model_kwargs(name, cfg, len(vocab), num_classes)
        model = build_model(name, kwargs, pretrained_embeddings=embeddings)
        display_name = MODEL_CLASSES[name].__name__

        logger.info("[DL] Training %s ...", display_name)
        start = time.perf_counter()
        val_metrics, best_epoch = fit(model, train_loader, val_loader, cfg, selection_metric, device)
        elapsed = time.perf_counter() - start
        logger.info(
            "[DL] %s | best epoch %d | val %s=%.4f | %.1fs",
            display_name, best_epoch, selection_metric, val_metrics[selection_metric], elapsed,
        )

        results.append(
            ModelResult(
                name=display_name,
                family="dl",
                estimator=DLPredictor(name, model, kwargs, vocab, max_len, device),
                params={
                    **kwargs,
                    "max_len": max_len,
                    "best_epoch": best_epoch,
                    "learning_rate": cfg.learning_rate,
                    "batch_size": cfg.batch_size,
                },
                val_metrics=val_metrics,
                train_seconds=elapsed,
            )
        )
    return results
