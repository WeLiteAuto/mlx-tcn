"""Training utilities for MLX-TCN models."""

from .config import TrainingConfig
from .data import DatasetBundle, DatasetMetadata, WaveformDataset, iter_batches, load_datasets
from .loop import EpochMetrics, EpochResult, run_training_loop, setup_optimizer
from .model import build_model
from .visualize import plot_history

__all__ = [
    "TrainingConfig",
    "DatasetBundle",
    "DatasetMetadata",
    "WaveformDataset",
    "EpochMetrics",
    "EpochResult",
    "build_model",
    "iter_batches",
    "load_datasets",
    "run_training_loop",
    "setup_optimizer",
    "plot_history",
]
