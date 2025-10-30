"""Unit tests for the ``weight_norm`` parametrisation utilities."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlx_tcn.parametrizations import weight_norm, remove_weight_norm
from mlx_tcn.conv import TemporalConv1d


def _collect_parameter_names(module) -> set[str]:
    """Return the flattened parameter names registered on a module."""
    names = set()

    def _walk(params: dict, prefix: str = ""):
        for key, value in params.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _walk(value, full)
            else:
                names.add(full)

    _walk(module.parameters())
    return names


def _random_input(batch: int, length: int, channels: int) -> mx.array:
    return mx.random.normal((batch, length, channels))


# ============================================================================
# weight_norm basic behaviour
# ============================================================================


def test_weight_norm_attaches_g_v_and_marks_module():
    conv = TemporalConv1d(32, 64, kernel_size=3, stride=1, dilation=1, causal=True)

    conv = weight_norm(conv, name="weight", dim=0)

    params = _collect_parameter_names(conv)
    assert "weight_v" in params
    assert "weight_g" in params
    assert "weight" in params  # materialised normalised weight

    assert hasattr(conv, "_weight_norm_params")
    assert "weight" in conv._weight_norm_params
    meta = conv._weight_norm_params["weight"]
    assert meta["dim"] == 0


def test_weight_norm_forward_matches_manual_normalisation():
    conv = TemporalConv1d(8, 16, kernel_size=3, stride=1, dilation=1, causal=True)

    x = _random_input(batch=2, length=10, channels=8)

    conv = weight_norm(conv, dim=0)

    out = conv(x)
    assert out.shape == (2, 10, 16)
    assert mx.all(mx.isfinite(out))

    # Manual reconstruction from stored parameters
    v = getattr(conv, "weight_v")
    g = getattr(conv, "weight_g")
    v_norm = mx.linalg.norm(v, axis=0, keepdims=True)
    manual_weight = g * (v / (v_norm + 1e-8))
    assert mx.allclose(conv.weight, manual_weight)


def test_weight_norm_is_deterministic():
    conv = TemporalConv1d(4, 6, kernel_size=3, stride=1, dilation=1, causal=True)
    conv = weight_norm(conv)

    x = _random_input(batch=3, length=7, channels=4)

    y1 = conv(x)
    y2 = conv(x)
    assert mx.allclose(y1, y2)


def test_weight_norm_multiple_forward_calls():
    conv = TemporalConv1d(16, 32, kernel_size=3, stride=1, dilation=1, causal=True)
    conv = weight_norm(conv)

    for _ in range(5):
        x = _random_input(batch=4, length=12, channels=16)
        out = conv(x)
        assert out.shape == (4, 12, 32)
        assert mx.all(mx.isfinite(out))


# ============================================================================
# remove_weight_norm behaviour
# ============================================================================


def test_remove_weight_norm_restores_parameters():
    conv = TemporalConv1d(10, 20, kernel_size=3, stride=1, dilation=1, causal=True)
    conv = weight_norm(conv)

    params_before = _collect_parameter_names(conv)
    assert "weight_v" in params_before
    assert "weight_g" in params_before

    conv = remove_weight_norm(conv)

    params_after = _collect_parameter_names(conv)
    assert "weight_v" not in params_after
    assert "weight_g" not in params_after
    assert "weight" in params_after

    assert not hasattr(conv, "_weight_norm_params")


# ============================================================================
# Error handling
# ============================================================================


def test_weight_norm_raises_if_parameter_missing():
    class Dummy:
        def __init__(self):
            self.bias = mx.zeros((4,))

        def parameters(self):
            return {"bias": self.bias}

    dummy = Dummy()

    with pytest.raises(ValueError, match="parameter 'weight'"):
        weight_norm(dummy)


def test_weight_norm_raises_if_applied_twice():
    conv = TemporalConv1d(4, 4, kernel_size=3, stride=1, dilation=1, causal=True)
    weight_norm(conv)

    with pytest.raises(ValueError, match="already applied"):
        weight_norm(conv)


def test_weight_norm_supports_negative_dim():
    conv = TemporalConv1d(4, 4, kernel_size=3, stride=1, dilation=1, causal=True)
    conv = weight_norm(conv, dim=-1)

    meta = conv._weight_norm_params["weight"]
    assert meta["dim"] == len(conv.weight.shape) - 1


def test_remove_weight_norm_errors_when_not_applied():
    conv = TemporalConv1d(3, 5, kernel_size=3, stride=1, dilation=1, causal=True)

    with pytest.raises(ValueError, match="not applied"):
        remove_weight_norm(conv)

