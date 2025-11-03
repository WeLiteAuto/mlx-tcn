# MLX Temporal Convolutional Networks

A re-implementation of the Temporal Convolutional Network (TCN) architecture in [Apple MLX](https://github.com/ml-explore/mlx). This project mirrors the design of popular PyTorch TCN toolkits—dilated causal convolutions, residual blocks, and skip connections—while adding MLX-specific conveniences such as streaming buffers and lightweight initialization helpers.

<p align="center">
  <img src="https://raw.githubusercontent.com/paul-krug/pytorch-tcn/main/assets/tcn_architecture.png" alt="Temporal Convolutional Network" width="600">
</p>

> **Reference:** Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. (ICLR 2018)

## Highlights

- Stacks of dilated causal convolutions for long receptive fields without pooling.
- Residual `TemporalBlock`s supporting layer/batch normalization, dropout, gated activations, and optional embeddings (`concat` or `add`).
- **Squeeze-and-Excitation (SE) blocks** for adaptive channel recalibration with configurable reduction ratios and residual connections.
- Optional global skip connections with output projection layers.
- Streaming-aware inference via `TemporalPad1d` and `BufferIO`, with extended buffer management utilities.
- Drop-in TCN module mirroring a PyTorch-style API, but using MLX arrays and modules.
- Inspired by [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn) while intentionally diverging in several behaviors (see "Differences from PyTorch-TCN" below).

## Project Status

- **Status:** Production ready (October 30, 2025)
- **Test Coverage:** 304 unit tests across all components (100% pass rate)
- **Key Capabilities:** Streaming inference (batch size 1), embedding fusion (`add`/`concat`), SE blocks for channel attention, global skip connections, configurable dilations and activations
- **Limitations:** Streaming currently limited to batch size 1; padding modes restricted to `zeros` and `replicate`; transposed conv requires `kernel_size = 2 * stride`

### Component Overview

- `buffer.py / BufferIO` – Iterator-style stream buffer manager for causal inference
- `pad.py / TemporalPad1d` – Causal & non-causal padding with buffer management
- `conv.py / TemporalConv1d` – Auto-padding temporal convs with streaming support
- `conv.py / TemporalConvTransposed1d` – Upsampling counterpart for decoder stacks
- `se.py / SqueezeExcitation` – Channel attention via global pooling and gating
- `tcn.py / TemporalBlock` – Residual blocks with normalization, gating, embeddings, SE
- `tcn.py / TCN` – Stackable architecture with dilations, skip connections, projections
- `modules.py / ModuleList` – MLX-compatible container mirroring PyTorch API

### Test Coverage

- **Total tests:** 304 (100% pass rate)
- **Module breakdown:**
  - `test_buffer.py`: 9 tests
  - `test_pad.py`: 25 tests
  - `test_conv.py`: 37 tests
  - `test_tcn.py`: 117 tests (including 45 SE layer tests)
  - `test_config.py`: 31 tests
  - `test_data.py`: 31 tests
  - `test_loop.py`: 30 tests
  - `test_model.py`: 20 tests
  - `test_visualize.py`: 4 tests

## Installation

Clone the repository and install dependencies (Apple Silicon is required for MLX):

```bash
git clone https://github.com/aaronge/mlx-tcn.git
cd mlx-tcn
pip install mlx pytest
```

If MLX is already present, you can skip reinstalling it.

## Getting started

### Constructing a TCN

```python
import mlx.core as mx
from mlx_tcn import TCN

model = TCN(
    num_inputs=64,
    num_channels=[128, 128, 256],
    kernel_sizes=3,
    dropout=0.1,
    causal=True,
    use_skip_connections=True,
    embedding_shapes=None,
    embedding_mode="add",
)

x = mx.random.normal((8, 200, 64))
logits = model(x, embeddings=None)
```

### Conditioning with embeddings

Each `TemporalBlock` can fuse auxiliary features. Supply a list of expected shapes to `embedding_shapes` and feed the corresponding tensors at call time.

```python
model = TCN(
    num_inputs=32,
    num_channels=[64, 64],
    kernel_sizes=5,
    embedding_shapes=[(16,), (8,)],
    embedding_mode="concat",
)

embeddings = [
    mx.random.normal((8, 16)),        # broadcast across time
    mx.random.normal((8, 200, 8)),    # time-aligned
]
logits = model(x, embeddings=embeddings)
```

### Using Squeeze-and-Excitation blocks

SE blocks perform adaptive channel recalibration through three steps:
1. **Squeeze** – Global average pooling compresses spatial information
2. **Excitation** – Two-layer MLP learns channel-wise importance 
3. **Scale** – Sigmoid-gated weights rescale feature maps

Enable them with `use_se=True`:

```python
model = TCN(
    num_inputs=32,
    num_channels=[64, 128, 256],
    kernel_sizes=3,
    causal=True,
    use_se=True,              # Enable SE blocks
    se_reduction=8,           # Reduction ratio for bottleneck (default: 8)
    se_residual=False,        # Add residual connection inside SE (default: False)
)

x = mx.random.normal((4, 100, 32))
out = model(x, inference=True)
```

The SE block squeezes spatial information via global pooling, learns channel importance through a two-layer MLP, and rescales the features. You can combine SE with other features like gated activations and embeddings:

```python
model = TCN(
    num_inputs=32,
    num_channels=[64, 128],
    kernel_sizes=3,
    use_se=True,
    se_reduction=4,
    use_gate=True,             # Combine SE with gated activations
    embedding_shapes=[(16,)],  # And embeddings
    embedding_mode="add",
)
```

**SE Parameter Guide:**

| `se_reduction` | Use Case | Parameters | Computation |
|----------------|----------|------------|-------------|
| 2 | High capacity | Most | Highest |
| 4 | Balanced | More | Higher |
| **8** (default) | **Recommended** | **Medium** | **Medium** |
| 16 | Lightweight | Least | Lowest |

**Parameter formula:** For a layer with `C` channels, SE adds approximately `2 × C × (C / reduction)` parameters.

**Example:** With `channels=128` and `reduction=8`, SE adds ~4K parameters (128×16 + 16×128).

### Streaming inference

Enable `causal=True`, reset internal buffers, and optionally manage padding state yourself via `BufferIO` when chunking the input sequence.

```python
from mlx_tcn import BufferIO

model.reset_buffers()
buffers = [BufferIO() for _ in range(len(model.network) * 2)]

chunk = mx.random.normal((1, 25, 64))
stream_out = model(chunk, embeddings=None, inference=True, in_buffer=buffers)
```

## Training tips

- Use causal convolutions (`causal=True`) for autoregressive tasks. Non-causal mode (default) behaves like same-padding convolutions for sequence tagging.
- Adjust dilation growth through `dilations` or `dilation_reset` to control receptive field size.
- The `kernel_initilaizer` argument accepts keys from `mlx_tcn/utils.py`—e.g. `"xavier_normal"`, `"he_uniform"`.
- When enabling skip connections, the model will project per-block outputs to the last channel width and sum them before the final activation.

**Using SE blocks effectively:**
- Start with `se_reduction=8` (default) for most use cases. Adjust based on your channel dimensions and parameter budget.
- Smaller reductions (e.g., 4) add more capacity but increase computation; larger reductions (e.g., 16) are more lightweight.
- SE blocks work well with classification tasks and when model capacity allows. Use cautiously in extremely lightweight models.
- Combine SE with other features: SE + gated activations (GLU) and SE + batch/layer normalization are effective combinations.
- SE adds <5% parameters typically but can significantly improve feature quality through channel recalibration.

## Tests

Run the unit tests with:

```bash
python -m pytest tests/unit
```

---

## Differences from PyTorch-TCN

This implementation borrows heavily from [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn) but is **not** a drop-in replacement. Keep the following divergences in mind:

- **Padding & buffer state** – Only `zeros`/`replicate` padding modes are supported, buffers live in shape `(1, pad_len, channels)` and are not registered as module buffers. The PyTorch code offers `zeros`/`reflect`/`replicate`/`circular`, stores buffers as `(1, channels, pad_len)`, and registers them for state dict I/O (`mlx_tcn/pad.py`; `_ref_pytorch_tcn/pytorch_tcn/pad.py`).
- **Normalization & weight norm** – `use_norm` accepts `'batch_norm'`, `'layer_norm'`, `'weight_norm'`, or `None`. The MLX port reparameterises convolution weights through `mlx_tcn.parametrizations.weight_norm`, yet MLX lacks PyTorch-style descriptor hooks—accessing `module.weight` shows the last cached tensor until another forward and the optional `remove_weight_norm` helper must be invoked manually. Behaviour is close, but still not a perfect drop-in (`mlx_tcn/tcn.py`, `mlx_tcn/parametrizations.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).
- **Tensor layout & GLU axis** – Inputs are `(batch, length, channels)` throughout, so `LayerNorm` runs channel-last and gated activations use `axis=-1`. PyTorch-TCN works in `(batch, channels, length)`, exposes an `input_shape` flag, and applies `nn.GLU(dim=1)` (`mlx_tcn/tcn.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).
- **Embedding shapes** – Embedding tensors are expected as `[batch, length, features]` (or broadcastable rank-2) and projected with channel-last convs. The PyTorch version expects channel-first shapes and validates the second dimension instead, so conditioning inputs must be reshaped when porting (`mlx_tcn/tcn.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).
- **Initialisers** – The MLX port exposes `he_*`, `xavier_*`, `normal`, `uniform` initialisers via `mlx_tcn/init.py`; PyTorch-TCN offers `kaiming_*` as well. Gain calculation semantics therefore differ when matching paper defaults (`mlx_tcn/tcn.py`, `mlx_tcn/init.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).
- **Streaming buffer helpers** – The legacy `get_in_buffers` convenience is commented out and `TemporalBlock` works with a pair of `BufferIO` objects per layer. PyTorch still ships `get_in_buffers` and slices flat buffer lists, although that branch now mainly serves backward compatibility (`mlx_tcn/tcn.py`, `mlx_tcn/buffer.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`, `_ref_pytorch_tcn/pytorch_tcn/buffer.py`).
- **Defaults & argument names** – MLX defaults to `causal=False`, accepts per-layer `kernel_sizes`, uses the misspelled `kernel_initilaizer`, and rejects unsupported options with hard errors. PyTorch-TCN defaults to `causal=True`, exposes `input_shape`, allows optional `weight_norm`, and retains legacy parameters such as `lookahead`/`in_buffers` for compatibility (`mlx_tcn/tcn.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).
- **Module registration** – PyTorch relies on `nn.ModuleList` so every block is automatically tracked; the MLX implementation stores layers in a Python list and relies on MLX’s attribute traversal, which can diverge when converting checkpoints or tooling that inspects module hierarchies (`mlx_tcn/tcn.py`; `_ref_pytorch_tcn/pytorch_tcn/tcn.py`).

Documenting these differences helps avoid surprises when porting training scripts or checkpoints from the PyTorch ecosystem.

## Project layout

```
mlx-tcn/
├── mlx_tcn/
│   ├── __init__.py
│   ├── buffer.py          # BufferIO for streaming chunk management
│   ├── conv.py            # TemporalConv1d / TemporalConvTransposed1d
│   ├── init.py            # MLX-friendly calculate_gain helper
│   ├── parametrizations.py # Weight normalization utilities
│   ├── pad.py             # TemporalPad1d with causal buffer support
│   ├── pool.py            # Adaptive pooling (AdaptiveAvgPool1d, AdaptiveMaxPool1d)
│   ├── se.py              # SqueezeExcitation channel attention
│   └── tcn.py             # TemporalBlock and TCN definitions
├── train/                 # Training pipeline components
│   ├── config.py          # TrainingConfig dataclass
│   ├── data.py            # Dataset loading and preprocessing
│   ├── model.py           # Model factory function
│   ├── loop.py            # Training and evaluation loops
│   ├── cli.py             # Command-line interface
│   └── visualize.py       # Training metrics visualization
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── test_buffer.py
│       ├── test_conv.py
│       ├── test_pad.py
│       ├── test_parametrizations.py
│       ├── test_tcn.py      # Includes 45 SE layer tests
│       ├── test_config.py
│       ├── test_data.py
│       ├── test_loop.py
│       ├── test_model.py
│       └── test_visualize.py
├── example.py             # Usage examples
└── README.md
```

## References

1. **Original TCN Paper**: Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. ICLR 2018.

2. **Squeeze-and-Excitation Networks**: Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.

3. **PyTorch-TCN**: https://github.com/paul-krug/pytorch-tcn

4. **MLX Framework**: https://github.com/ml-explore/mlx

## License & Acknowledgements

This project follows the spirit of the PyTorch TCN implementations while targeting MLX. The included architectural diagram comes from the PyTorch TCN repository by Paul Krug. If you build something with this codebase, consider citing the original TCN paper linked above.
