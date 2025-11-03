"""Squeeze-and-Excitation utilities for temporal convolution blocks."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Lightweight Squeeze-and-Excitation (SE) block for MLX tensors.

    This module supports inputs of shape ``(batch, length, channels)`` or
    ``(batch, channels)``. It performs global average pooling across the
    temporal dimension, applies a bottleneck MLP, and returns channel-wise
    re-scaled features.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the hidden layer. A value of ``0``
            keeps the hidden dimensionality equal to ``channels``.
        residual: If ``True``, adds a residual connection from the input to
            the scaled output.
        apply_to_input: If ``False``, only the excitation weights are
            returned without scaling the input tensor.
    """

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        residual: bool = False,
        apply_to_input: bool = True,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError("`channels` must be a positive integer.")
        if reduction < 0:
            raise ValueError("`reduction` must be non-negative.")

        reduced_channels = channels if reduction == 0 else max(1, channels // reduction)

        self.reduce: Optional[nn.Linear]
        if reduction != 0:
            self.reduce = nn.Linear(channels, reduced_channels, bias=True)
        else:
            self.reduce = None

        self.expand = nn.Linear(reduced_channels, channels, bias=True)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()
        self.residual = residual
        self.apply_to_input = apply_to_input

    def _squeeze(self, x: mx.array) -> mx.array:
        # Global average pooling across the temporal dimension.
        return mx.mean(x, axis=1, keepdims=True)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim not in (2, 3):
            raise ValueError(
                "SE block expects a tensor shaped as (B, L, C) or (B, C); "
                f"received array with shape {x.shape}."
            )

        squeeze_only = x.ndim == 2
        if squeeze_only:
            x = x[:, None, :]

        weights = self._squeeze(x)

        if self.reduce is not None:
            weights = self.activation(self.reduce(weights))

        weights = self.gate(self.expand(weights))

        output = weights
        if self.apply_to_input:
            output = x * weights
            if self.residual:
                output = output + x

        if squeeze_only:
            output = output[:, 0, :]

        return output

