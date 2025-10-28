from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import TrainingConfig
from .data import DatasetBundle, WaveformDataset, iter_batches


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    mae: float


@dataclass(slots=True)
class EpochResult:
    epoch: int
    train: EpochMetrics
    val: EpochMetrics


def setup_optimizer(config: TrainingConfig) -> object:
    """Create an Adam optimiser configured from the training config."""
    if config.weight_decay and config.weight_decay > 0.0:
        try:
            return optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        except AttributeError as exc:
            raise RuntimeError(
                "AdamW optimiser is not available in the installed MLX version. "
                "Set weight_decay=0 or upgrade MLX."
            ) from exc
    return optim.Adam(learning_rate=config.learning_rate)


def _select_sequence_logits(logits: mx.array) -> mx.array:
    """Reduce the temporal dimension by selecting the final timestep logits."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits to have shape (batch, time, classes). Got {logits.shape}")
    return logits[:, -1, :]


def create_train_step(optimizer):
    """Build a training step that applies gradients and returns metrics."""

    def loss_only(model, x, y):
        logits = model(x, embeddings=None, inference=False)
        pooled = _select_sequence_logits(logits)
        return nn.losses.mse_loss(pooled, y)

    value_and_grad = mx.value_and_grad(loss_only)

    def step(model, x, y):
        loss, grads = value_and_grad(model, x, y)
        optimizer.update(model, grads)
        logits = model(x, embeddings=None, inference=False)
        pooled_logits = _select_sequence_logits(logits)
        return loss, pooled_logits

    return step


def _sum_absolute_error(predictions: mx.array, targets: mx.array) -> mx.array:
    return mx.sum(mx.abs(predictions - targets))


def train_epoch(
    model: nn.Module,
    dataset: WaveformDataset,
    train_step,
    config: TrainingConfig,
    epoch: int,
) -> EpochMetrics:
    model.train()

    total_loss = 0.0
    total_abs_error = 0.0
    total_examples = 0
    total_targets = 0

    iterator_seed = config.seed + epoch
    for step, (x_batch, y_batch, cont_batch, disc_batch) in enumerate(
        iter_batches(dataset, config.batch_size, shuffle=True, seed=iterator_seed)
    ):
        loss, pooled_logits = train_step(model, x_batch, y_batch)
        mx.eval(loss, pooled_logits)

        batch_size = y_batch.shape[0]
        total_loss += loss.item() * batch_size

        abs_err = _sum_absolute_error(pooled_logits, y_batch)
        mx.eval(abs_err)
        total_abs_error += float(abs_err.item())
        total_examples += batch_size
        total_targets += batch_size * y_batch.shape[1]

        if (step + 1) % max(config.log_every_n_steps, 1) == 0:
            print(f"[epoch {epoch}] step {step + 1}: loss={loss.item():.4f}")

    if total_examples == 0:
        raise RuntimeError("Training dataset produced zero samples; check the data loader.")

    avg_loss = total_loss / total_examples
    avg_mae = total_abs_error / total_targets
    return EpochMetrics(avg_loss, avg_mae)


def evaluate(model, dataset: WaveformDataset, config: TrainingConfig) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_examples = 0
    total_targets = 0

    for x_batch, y_batch, cont_batch, disc_batch in iter_batches(
        dataset, config.batch_size, shuffle=False
    ):
        logits = model(x_batch, embeddings=None, inference=False)
        pooled = _select_sequence_logits(logits)
        loss = nn.losses.mse_loss(pooled, y_batch)
        mx.eval(loss, pooled)

        batch_size = y_batch.shape[0]
        total_loss += loss.item() * batch_size

        abs_err = _sum_absolute_error(pooled, y_batch)
        mx.eval(abs_err)
        total_abs_error += float(abs_err.item())
        total_examples += batch_size
        total_targets += batch_size * y_batch.shape[1]

    if total_examples == 0:
        return EpochMetrics(loss=0.0, mae=0.0)

    avg_loss = total_loss / total_examples
    avg_mae = total_abs_error / total_targets
    return EpochMetrics(avg_loss, avg_mae)


def run_training_loop(model, datasets: DatasetBundle, config: TrainingConfig) -> List[EpochResult]:
    optimizer = setup_optimizer(config)
    try:
        optimizer.init(model.trainable_parameters())
    except (AttributeError, TypeError) as exc:  # pragma: no cover - optional init
        print(f"Warning: optimizer init failed ({exc}); continuing without explicit init.")
    train_step = create_train_step(optimizer)

    history: List[EpochResult] = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_epoch(
            model=model,
            dataset=datasets.train,
            train_step=train_step,
            config=config,
            epoch=epoch,
        )
        val_metrics = evaluate(model, datasets.val, config)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics.loss:.4f} train_mae={train_metrics.mae:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_mae={val_metrics.mae:.4f}"
        )

        history.append(EpochResult(epoch=epoch, train=train_metrics, val=val_metrics))

    return history
