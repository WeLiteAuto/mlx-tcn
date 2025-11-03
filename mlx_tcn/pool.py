"""Pooling utilities shared across training and library usage."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

OutputSizeArg = Union[int, Sequence[Optional[int]], None]


def _normalise_output_size(spec: OutputSizeArg) -> Optional[int]:
    if spec is None:
        return None

    if isinstance(spec, (list, tuple)):
        if len(spec) != 1:
            raise ValueError(
                "Adaptive 1D pooling expects a single output size; "
                f"received {len(spec)} entries."
            )
        spec = spec[0]

    if spec is None:
        return None

    if not isinstance(spec, int):
        raise TypeError(
            "Output size must be an integer, `None`, or a length-1 sequence "
            f"thereof (got {type(spec)})."
        )

    if spec <= 0:
        raise ValueError("Output size must be a positive integer.")

    return spec


def _validate_input(x: mx.array) -> Tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(
            "Adaptive 1D pooling modules expect inputs shaped as (B, L, C); "
            f"received array with {x.ndim} dimensions."
        )

    return x.shape  # type: ignore[return-value]


def _adaptive_pool1d(
    x: mx.array,
    output_size: int,
    reducer: Callable[[mx.array], mx.array],
) -> mx.array:
    batch, length, channels = x.shape

    pooled: List[mx.array] = []
    for idx in range(output_size):
        start = (idx * length) // output_size
        end = ((idx + 1) * length + output_size - 1) // output_size

        if end <= start:
            raise RuntimeError(
                "Invalid pooling window: start index is not smaller than the end "
                f"index (start={start}, end={end})."
            )

        window = x[:, start:end, :]
        pooled.append(reducer(window))

    return mx.concatenate(pooled, axis=1) if pooled else x[:, :0, :]


class AdaptiveAvgPool1d(nn.Module):
    """
    MLX implementation mirroring :class:`torch.nn.AdaptiveAvgPool1d`.

    Parameters
    ----------
    output_size:
        Target sequence length. Accepts an integer, ``None`` (keep original
        length), or a length-1 sequence containing either option.
    """

    def __init__(self, output_size: OutputSizeArg):
        super().__init__()
        self.output_size = _normalise_output_size(output_size)

    def __call__(self, x: mx.array) -> mx.array:
        _, length, _ = _validate_input(x)
        target_length = length if self.output_size is None else self.output_size

        def reducer(window: mx.array) -> mx.array:
            return mx.mean(window, axis=1, keepdims=True)

        return _adaptive_pool1d(x, target_length, reducer)

    def __repr__(self) -> str:
        size = self.output_size if self.output_size is not None else "None"
        return f"{self.__class__.__name__}(output_size={size})"


class AdaptiveMaxPool1d(nn.Module):
    """
    MLX implementation mirroring :class:`torch.nn.AdaptiveMaxPool1d`.

    Parameters
    ----------
    output_size:
        Target sequence length. Accepts an integer, ``None`` (keep original
        length), or a length-1 sequence containing either option.
    return_indices:
        If ``True``, also returns the indices of maximal elements.
    """

    def __init__(self, output_size: OutputSizeArg, *, return_indices: bool = False):
        super().__init__()
        self.output_size = _normalise_output_size(output_size)
        self.return_indices = return_indices

    def __call__(self, x: mx.array) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        _, length, _ = _validate_input(x)
        target_length = length if self.output_size is None else self.output_size

        def reducer(window: mx.array) -> mx.array:
            return mx.max(window, axis=1, keepdims=True)

        values = _adaptive_pool1d(x, target_length, reducer)

        if not self.return_indices:
            return values

        indices: List[mx.array] = []
        for idx in range(target_length):
            start = (idx * length) // target_length
            end = ((idx + 1) * length + target_length - 1) // target_length
            window = x[:, start:end, :]
            local_idx = mx.argmax(window, axis=1)
            if local_idx.dtype != mx.int32:
                local_idx = local_idx.astype(mx.int32)
            global_idx = local_idx + start
            indices.append(global_idx[:, mx.newaxis, :])

        return values, mx.concatenate(indices, axis=1)

    def __repr__(self) -> str:
        size = self.output_size if self.output_size is not None else "None"
        return (
            f"{self.__class__.__name__}(output_size={size}, "
            f"return_indices={self.return_indices})"
        )


__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveMaxPool1d",
    "_adaptive_pool1d",
    "_normalise_output_size",
    "_validate_input",
    "OutputSizeArg",
]
