from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .loop import EpochResult


def plot_history(
    history: Sequence[EpochResult],
    output_path: Path,
    show: bool = False,
) -> None:
    """Plot training/validation loss and MAE curves using Matplotlib."""
    if not history:
        print("No history to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for plotting but is not installed. "
            "Install it with `pip install matplotlib`."
        ) from exc

    epochs = [result.epoch for result in history]
    train_losses = [result.train.loss for result in history]
    val_losses = [result.val.loss for result in history]
    train_mae = [result.train.mae for result in history]
    val_mae = [result.val.mae for result in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax_loss.plot(epochs, val_losses, label="Val Loss", marker="o")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend()

    ax_mae = axes[1]
    ax_mae.plot(epochs, train_mae, label="Train MAE", marker="o")
    ax_mae.plot(epochs, val_mae, label="Val MAE", marker="o")
    ax_mae.set_title("Mean Absolute Error")
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, linestyle="--", alpha=0.4)
    ax_mae.legend()

    fig.suptitle("MLX-TCN Training Curves")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
