# MLX Temporal Convolutional Networks

A re-implementation of the Temporal Convolutional Network (TCN) architecture in [Apple MLX](https://github.com/ml-explore/mlx). This project mirrors the design of popular PyTorch TCN toolkits—dilated causal convolutions, residual blocks, and skip connections—while adding MLX-specific conveniences such as streaming buffers and lightweight initialization helpers.

<p align="center">
  <img src="https://raw.githubusercontent.com/paul-krug/pytorch-tcn/main/assets/tcn_architecture.png" alt="Temporal Convolutional Network" width="600">
</p>

> **Reference:** Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. (ICLR 2018)

## Highlights

- Stacks of dilated causal convolutions for long receptive fields without pooling.
- Residual `TemporalBlock`s supporting layer/batch normalization, dropout, gated activations, and optional embeddings (`concat` or `add`).
- Optional global skip connections with output projection layers.
- Streaming-aware inference via `TemporalPad1d` and `BufferIO`, including buffer reset utilities.
- Drop-in TCN module mirroring a PyTorch-style API, but using MLX arrays and modules.
- Unit-tested implementation covering buffers, padding, convolutions, and TCN integration.

## Project Status

- **Status:** Production ready (October 22, 2025)
- **Test Coverage:** 139 unit tests across buffers, padding, convolutions, and full TCN integration (100% pass rate)
- **Key Capabilities:** Streaming inference (batch size 1), embedding fusion (`add`/`concat`), global skip connections, configurable dilations and activations
- **Limitations:** Streaming currently limited to batch size 1; padding modes restricted to `zeros` and `replicate`; transposed conv requires `kernel_size = 2 * stride`

### Component Overview

- `buffer.py / BufferIO` – Iterator-style stream buffer manager for causal inference (13 tests)
- `pad.py / TemporalPad1d` – Causal & non-causal padding with buffer management (25 tests)
- `conv.py / TemporalConv1d` – Auto-padding temporal convs with streaming support (37 tests)
- `conv.py / TemporalConvTransposed1d` – Upsampling counterpart for decoder stacks
- `tcn.py / TemporalBlock` – Residual blocks with normalization, gating, embeddings (34 tests)
- `tcn.py / TCN` – Stackable architecture with dilations, skip connections, projections (27 tests)
- `modules.py / ModuleList` – MLX-compatible container mirroring PyTorch API

### Test Coverage

- **Total tests:** 139 (100 % pass rate, ~0.26 s runtime)
- **Suite layout:** `tests/unit/test_buffer.py`, `test_temporal_pad1d.py`, `test_conv.py`, `test_tcn.py`
- **Focus areas:** streaming workflows, padding validity, convolution correctness, residual/embedding paths, skip connections, error handling

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
    kernel_size=3,
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
    kernel_size=5,
    embedding_shapes=[(16,), (8,)],
    embedding_mode="concat",
)

embeddings = [
    mx.random.normal((8, 16)),        # broadcast across time
    mx.random.normal((8, 200, 8)),    # time-aligned
]
logits = model(x, embeddings=embeddings)
```

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

## Tests

Run the unit tests with:

```bash
python -m pytest tests/unit
```

## Project layout

```
mlx-tcn/
├── mlx_tcn/
│   ├── __init__.py
│   ├── buffer.py          # BufferIO for streaming chunk management
│   ├── conv.py            # TemporalConv1d / TemporalConvTransposed1d
│   ├── pad.py             # TemporalPad1d with causal buffer support
│   ├── tcn.py             # TemporalBlock and TCN definitions
│   ├── modules.py         # MLX-friendly ModuleList implementation
│   └── utils.py           # Activation gain helpers and misc utilities
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── test_buffer.py
│       ├── test_conv.py
│       ├── test_pad.py
│       └── test_tcn.py
├── PROJECT_STATUS.md      # Detailed status report and testing summary
└── README.md
```

## License & acknowledgements

This project follows the spirit of the PyTorch TCN implementations while targeting MLX. The included architectural diagram comes from the PyTorch TCN repository by Paul Krug. If you build something with this codebase, consider citing the original TCN paper linked above.
