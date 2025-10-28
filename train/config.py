from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(slots=True)
class TrainingConfig:
    """Configuration container for the TCN training pipeline."""

    data_dir: Path = Path("data")
    input_file: str = "data_input.npz"
    label_file: str = "data_labels.npz"
    label_keys: Tuple[str, ...] = ("HIC", "Dmax", "Nij")

    train_split: float = 0.8
    seed: int = 13

    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    log_every_n_steps: int = 10

    num_channels: Tuple[int, ...] = (64, 64, 128)
    kernel_sizes: Union[int, Tuple[int, ...]] = 3
    dropout: float = 0.1
    causal: bool = False
    use_norm: str = "batch_norm"
    activation: str = "relu"
    use_skip_connections: bool = True
    use_gate: bool = False
    output_activation: Optional[str] = None
    enable_plot: bool = False
    plot_path: Optional[Path] = None

    def input_path(self) -> Path:
        return self.data_dir / self.input_file

    def label_path(self) -> Path:
        return self.data_dir / self.label_file

    def resolve_plot_path(self) -> Path:
        default_name = "training_metrics.png"
        return self.plot_path if self.plot_path is not None else Path(default_name)

    def with_updates(self, **kwargs) -> "TrainingConfig":
        """Return a copy of the config with the provided fields updated."""
        return replace(self, **kwargs)
