from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import mlx.core as mx
import numpy as np

from .config import TrainingConfig


@dataclass(slots=True)
class DatasetMetadata:
    num_features: int
    seq_len: int
    target_dim: int
    target_names: tuple[str, ...]
    num_continuous_params: int
    num_discrete_params: int
    discrete_indices: tuple[int, ...]


@dataclass(slots=True)
class WaveformDataset:
    features: np.ndarray
    labels: np.ndarray
    continuous_params: np.ndarray
    discrete_params: np.ndarray

    def __post_init__(self) -> None:
        if self.features.dtype != np.float32:
            self.features = self.features.astype(np.float32, copy=False)
        if self.labels.dtype != np.float32:
            self.labels = self.labels.astype(np.float32, copy=False)
        if self.continuous_params.dtype != np.float32:
            self.continuous_params = self.continuous_params.astype(np.float32, copy=False)
        if self.discrete_params.dtype != np.int32:
            self.discrete_params = self.discrete_params.astype(np.int32, copy=False)

    def __len__(self) -> int:
        return self.features.shape[0]

    def get_batch(
        self, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.features[indices],
            self.labels[indices],
            self.continuous_params[indices],
            self.discrete_params[indices],
        )


@dataclass(slots=True)
class DatasetBundle:
    train: WaveformDataset
    val: WaveformDataset
    metadata: DatasetMetadata


def load_datasets(config: TrainingConfig) -> DatasetBundle:
    """Load waveform data from disk, preprocess, and create train/validation splits."""
    with np.load(config.input_path()) as input_data:
        if "waveforms" not in input_data:
            raise KeyError("Expected key 'waveforms' in input npz file.")
        waveforms = input_data["waveforms"]
        if "params" not in input_data:
            raise KeyError("Expected key 'params' in input npz file.")
        params = input_data["params"]

    with np.load(config.label_path()) as label_data:
        missing = [key for key in config.label_keys if key not in label_data]
        if missing:
            available = ", ".join(sorted(label_data.keys()))
            raise KeyError(
                f"Missing target keys: {missing}. Available label keys: {available}"
            )
        labels_stack = [label_data[key] for key in config.label_keys]
        labels_raw = np.stack(labels_stack, axis=-1)

    if waveforms.ndim != 3:
        raise ValueError(f"Expected waveforms to have shape (N, C, T). Got {waveforms.shape}")

    if params.ndim != 2:
        raise ValueError(f"Expected params to have shape (N, F). Got {params.shape}")

    # Move channel dimension to the end for TCN consumption -> (N, T, C)
    features = np.transpose(waveforms, (0, 2, 1)).astype(np.float32, copy=False)

    labels = labels_raw.astype(np.float32, copy=False)

    discrete_indices = (3, 9, 11, 14)
    if max(discrete_indices) >= params.shape[1]:
        raise ValueError(
            f"Discrete feature index out of range for params of shape {params.shape}."
        )
    mask = np.ones(params.shape[1], dtype=bool)
    mask[list(discrete_indices)] = False
    params_continuous = params[:, mask].astype(np.float32, copy=False)
    params_discrete = params[:, list(discrete_indices)].astype(np.int32, copy=False)

    metadata = DatasetMetadata(
        num_features=features.shape[-1],
        seq_len=features.shape[1],
        target_dim=labels.shape[-1],
        target_names=tuple(config.label_keys),
        num_continuous_params=params_continuous.shape[1],
        num_discrete_params=len(discrete_indices),
        discrete_indices=discrete_indices,
    )

    num_samples = features.shape[0]
    if not (0.0 < config.train_split < 1.0):
        raise ValueError("train_split must be between 0 and 1 (exclusive).")

    split_idx = int(num_samples * config.train_split)
    if split_idx == 0 or split_idx == num_samples:
        raise ValueError("Train/validation split would produce an empty split. Adjust train_split.")

    rng = np.random.default_rng(config.seed)
    permutation = rng.permutation(num_samples)

    train_idx = permutation[:split_idx]
    val_idx = permutation[split_idx:]

    train_dataset = WaveformDataset(
        features[train_idx],
        labels[train_idx],
        params_continuous[train_idx],
        params_discrete[train_idx],
    )
    val_dataset = WaveformDataset(
        features[val_idx],
        labels[val_idx],
        params_continuous[val_idx],
        params_discrete[val_idx],
    )

    return DatasetBundle(train=train_dataset, val=val_dataset, metadata=metadata)


def iter_batches(
    dataset: WaveformDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
    """Yield batches of data converted to MLX arrays."""
    num_items = len(dataset)
    if num_items == 0:
        return

    indices = np.arange(num_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_indices = indices[start:end]
        batch_x, batch_y, batch_cont, batch_disc = dataset.get_batch(batch_indices)
        yield (
            mx.array(batch_x),
            mx.array(batch_y),
            mx.array(batch_cont),
            mx.array(batch_disc),
        )
