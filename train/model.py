from __future__ import annotations

from typing import Sequence

from mlx_tcn import TCN

from .config import TrainingConfig
from .data import DatasetMetadata


def build_model(config: TrainingConfig, metadata: DatasetMetadata) -> TCN:
    """Instantiate a TCN model parameterised by the training configuration."""
    kernel_sizes: Sequence[int] | int = config.kernel_sizes
    if isinstance(kernel_sizes, (tuple, list)):
        if len(kernel_sizes) != len(config.num_channels):
            raise ValueError(
                "kernel_sizes length must match num_channels "
                f"(got {len(kernel_sizes)} vs {len(config.num_channels)})"
            )
        kernel_sizes_arg = list(kernel_sizes)
    elif isinstance(kernel_sizes, int):
        kernel_sizes_arg = kernel_sizes
    else:
        raise TypeError("kernel_sizes must be an int or a sequence of ints.")

    model = TCN(
        num_inputs=metadata.num_features,
        num_channels=list(config.num_channels),
        kernel_sizes=kernel_sizes_arg,
        dropout=config.dropout,
        causal=config.causal,
        use_norm=config.use_norm,
        activation=config.activation,
        use_skip_connections=config.use_skip_connections,
        use_gate=config.use_gate,
        output_projection=metadata.target_dim,
        output_activation=config.output_activation,
    )
    return model
