from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx

from .config import TrainingConfig
from .data import load_datasets
from .loop import run_training_loop
from .model import build_model
from .visualize import plot_history


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLX-TCN model on waveform data.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing the input/label npz files.")
    parser.add_argument(
        "--label-keys",
        type=str,
        nargs="+",
        help="Label keys to load from labels npz (default: HIC Dmax Nij).",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument("--train-split", type=float, help="Train split ratio (between 0 and 1).")
    parser.add_argument("--log-steps", type=int, dest="log_every", help="Logging frequency in training steps.")
    parser.add_argument("--num-channels", type=int, nargs="+", help="Hidden channel sizes per block.")
    parser.add_argument("--kernel-sizes", type=int, nargs="+", help="Kernel sizes per block.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--causal", action="store_true", help="Enable causal convolutions.")
    parser.add_argument("--disable-skip-connections", action="store_true", help="Turn off global skip connections.")
    parser.add_argument("--use-gate", action="store_true", help="Enable gated activations in residual blocks.")
    parser.add_argument("--output-activation", type=str, help="Optional activation applied after output projection.")
    parser.add_argument("--plot", action="store_true", help="Plot training curves with Matplotlib.")
    parser.add_argument("--plot-file", type=Path, help="Path to save the training curve plot (PNG).")
    return parser.parse_args(argv)


def build_config_from_args(
    args: argparse.Namespace, defaults: Optional[TrainingConfig] = None
) -> TrainingConfig:
    base_config = defaults if defaults is not None else TrainingConfig()
    updates: Dict[str, Any] = {}

    if args.data_dir is not None:
        updates["data_dir"] = args.data_dir
    if args.label_keys is not None:
        updates["label_keys"] = tuple(args.label_keys)
    if args.epochs is not None:
        updates["num_epochs"] = args.epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        updates["learning_rate"] = args.learning_rate
    if args.train_split is not None:
        updates["train_split"] = args.train_split
    if args.log_every is not None:
        updates["log_every_n_steps"] = args.log_every
    if args.num_channels is not None:
        updates["num_channels"] = tuple(args.num_channels)
    if args.kernel_sizes is not None:
        updates["kernel_sizes"] = tuple(args.kernel_sizes)
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.causal:
        updates["causal"] = True
    if args.disable_skip_connections:
        updates["use_skip_connections"] = False
    if args.use_gate:
        updates["use_gate"] = True
    if args.output_activation is not None:
        updates["output_activation"] = args.output_activation
    if args.plot:
        updates["enable_plot"] = True
    if args.plot_file is not None:
        updates["plot_path"] = args.plot_file
        updates["enable_plot"] = True

    if updates:
        return base_config.with_updates(**updates)
    if defaults is None:
        return base_config
    return replace(base_config)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    config = build_config_from_args(args)

    mx.random.seed(config.seed)

    datasets = load_datasets(config)
    print(
        "Loaded dataset: "
        f"{len(datasets.train)} train / {len(datasets.val)} val samples, "
        f"{datasets.metadata.target_dim} targets ({', '.join(datasets.metadata.target_names)}), "
        f"sequence length {datasets.metadata.seq_len}, "
        f"{datasets.metadata.num_features} features."
    )

    model = build_model(config, datasets.metadata)
    history = run_training_loop(model, datasets, config)

    if history:
        final = history[-1]
        print(
            f"Finished training at epoch {final.epoch:03d}: "
            f"train_loss={final.train.loss:.4f}, train_mae={final.train.mae:.4f}, "
            f"val_loss={final.val.loss:.4f}, val_mae={final.val.mae:.4f}"
        )
        if config.enable_plot:
            output_path = config.resolve_plot_path()
            plot_history(history, output_path)


if __name__ == "__main__":
    main()
