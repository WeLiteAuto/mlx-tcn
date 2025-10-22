import os
import sys

import mlx.core as mx

tests_dir = os.path.dirname(__file__)
project_src = os.path.abspath(os.path.join(tests_dir, "..", "..", "mlx-tcn"))
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from pad import TemporalPad1d, PADDING_MODES  # noqa: E402
from buffer import BufferIO  # noqa: E402


class AssertionError_(Exception):
    """Custom exception for test assertion failures."""
    pass


def pytest_raises(exc_class, match=None):
    """Simple context manager to replace pytest.raises."""
    class RaisesContext:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError_(f"Expected {exc_class.__name__} but no exception was raised")
            if not issubclass(exc_type, exc_class):
                return False  # Re-raise the actual exception
            if match and match not in str(exc_val):
                raise AssertionError_(f"Exception message '{exc_val}' does not contain '{match}'")
            return True  # Suppress the expected exception
    
    return RaisesContext()


class pytest:
    raises = staticmethod(pytest_raises)


def assert_array_equal(actual, expected, rtol=1e-5, atol=1e-8):
    """Helper function to assert MLX arrays are equal."""
    # Convert to mx.array if needed
    actual_mx = actual if isinstance(actual, mx.array) else mx.array(actual)
    
    # Handle scalar expected values - broadcast to match actual shape
    if isinstance(expected, (int, float)):
        expected_mx = mx.full(actual_mx.shape, expected)
    else:
        expected_mx = expected if isinstance(expected, mx.array) else mx.array(expected)
    
    # Check shapes match
    assert actual_mx.shape == expected_mx.shape, \
        f"Shape mismatch: {actual_mx.shape} vs {expected_mx.shape}"
    
    # Check values are close
    assert mx.allclose(actual_mx, expected_mx, rtol=rtol, atol=atol).item(), \
        f"Arrays not equal:\nActual: {actual_mx}\nExpected: {expected_mx}\nDiff: {mx.abs(actual_mx - expected_mx)}"


def assert_array_almost_equal(actual, expected, decimal=6):
    """Helper function to assert MLX arrays are almost equal."""
    atol = 10.0 ** (-decimal)
    assert_array_equal(actual, expected, rtol=1e-5, atol=atol)


# ============================================================================
# Initialization Tests
# ============================================================================

def test_temporal_pad1d_init_with_none_buffer():
    """Test initialization with None buffer creates zeros."""
    pad_len = 4
    in_channels = 2
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)
    
    assert padder.pad_len == pad_len
    assert padder.buffer.shape == (1, pad_len, in_channels)
    assert_array_equal(padder.buffer, 0.0)


def test_temporal_pad1d_init_with_float_buffer():
    """Test initialization with float buffer creates filled array."""
    pad_len = 3
    in_channels = 2
    fill_value = 5.0
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=fill_value, causal=True)
    
    assert padder.buffer.shape == (1, pad_len, in_channels)
    assert_array_equal(padder.buffer, fill_value)


def test_temporal_pad1d_init_with_array_buffer():
    """Test initialization with mx.array buffer."""
    pad_len = 2
    in_channels = 3
    custom_buffer = mx.ones((1, pad_len, in_channels)) * 7.0
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=custom_buffer, causal=True)
    
    assert mx.array_equal(padder.buffer, custom_buffer)


def test_temporal_pad1d_init_invalid_buffer_type():
    """Test that invalid buffer type raises ValueError."""
    with pytest.raises(ValueError, match="buffer must be"):
        TemporalPad1d(padding=2, in_channels=1, buffer="invalid", causal=True)


def test_temporal_pad1d_init_invalid_padding_mode():
    """Test that invalid padding mode raises ValueError."""
    with pytest.raises(ValueError, match="padding_mode must be one of"):
        TemporalPad1d(padding=2, in_channels=1, buffer=None, padding_mode="invalid")


def test_temporal_pad1d_init_causal_left_padding():
    """Test causal mode sets left padding only."""
    pad_len = 5
    padder = TemporalPad1d(padding=pad_len, in_channels=1, buffer=None, causal=True)
    
    assert padder.left_padding == pad_len
    assert padder.right_padding == 0


def test_temporal_pad1d_init_non_causal_symmetric_padding():
    """Test non-causal mode splits padding symmetrically."""
    pad_len = 4
    padder = TemporalPad1d(padding=pad_len, in_channels=1, buffer=None, causal=False)
    
    assert padder.left_padding == 2
    assert padder.right_padding == 2


def test_temporal_pad1d_init_non_causal_odd_padding():
    """Test non-causal mode with odd padding."""
    pad_len = 5
    padder = TemporalPad1d(padding=pad_len, in_channels=1, buffer=None, causal=False)
    
    assert padder.left_padding == 2
    assert padder.right_padding == 3
    assert padder.left_padding + padder.right_padding == pad_len


# ============================================================================
# Offline Padding Tests (Training Mode)
# ============================================================================

def test_temporal_pad1d_offline_zero_padding():
    """Test offline zero padding in non-causal mode."""
    pad_len = 4
    in_channels = 2
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=False)

    x = mx.reshape(mx.arange(1 * 5 * in_channels, dtype=mx.float32), (1, 5, in_channels))
    padded = padder(x, inference=False)

    assert padded.shape == (1, 5 + pad_len, in_channels)

    assert_array_equal(padded[:, :pad_len // 2, :], 0.0)
    assert_array_equal(padded[:, -pad_len // 2:, :], 0.0)
    assert_array_equal(padded[:, pad_len // 2:-pad_len // 2, :], x)


def test_temporal_pad1d_offline_causal_zero_padding():
    """Test offline causal zero padding."""
    pad_len = 3
    in_channels = 1
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)

    x = mx.reshape(mx.arange(1 * 4 * in_channels, dtype=mx.float32), (1, 4, in_channels))
    padded = padder(x, inference=False)

    assert padded.shape == (1, 4 + pad_len, in_channels)

    assert_array_equal(padded[:, :pad_len, :], 0.0)
    assert_array_equal(padded[:, pad_len:, :], x)


def test_temporal_pad1d_offline_replicate_padding():
    """Test offline replicate (edge) padding mode."""
    pad_len = 2
    in_channels = 1
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None,
                          padding_mode="replicate", causal=False)

    x = mx.array([[[1.0], [2.0], [3.0], [4.0]]])
    padded = padder(x, inference=False)

    assert padded.shape == (1, 6, in_channels)
    padded_flat = mx.reshape(padded, (-1,))
    expected = mx.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])
    assert_array_almost_equal(padded_flat, expected)


def test_temporal_pad1d_offline_batch_size_greater_than_one():
    """Test offline padding works with batch size > 1."""
    pad_len = 2
    in_channels = 2
    batch_size = 3
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=False)

    x = mx.ones((batch_size, 5, in_channels))
    padded = padder(x, inference=False)

    assert padded.shape == (batch_size, 5 + pad_len, in_channels)


# ============================================================================
# Inference Padding Tests (Streaming Mode)
# ============================================================================

def test_temporal_pad1d_causal_inference_updates_buffer():
    """Test streaming inference with causal padding updates internal buffer."""
    pad_len = 3
    in_channels = 1
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)

    x = mx.reshape(mx.arange(1 * 4 * in_channels, dtype=mx.float32) + 1, (1, 4, in_channels))
    padded = padder(x, inference=True)

    assert padded.shape == (1, 4 + pad_len, in_channels)

    assert_array_equal(padded[:, :pad_len, :], 0.0)
    assert_array_equal(padded[:, pad_len:, :], x)

    expected_buffer = padded[:, -pad_len:, :]
    assert_array_equal(padder.buffer, expected_buffer)


def test_temporal_pad1d_inference_non_causal_raises():
    """Test that inference with non-causal mode raises ValueError."""
    padder = TemporalPad1d(padding=2, in_channels=1, buffer=None, causal=False)
    x = mx.ones((1, 4, 1))
    
    with pytest.raises(ValueError, match="Causal padding is not supported"):
        padder(x, inference=True)


def test_temporal_pad1d_inference_batch_size_not_one_raises():
    """Test that inference with batch size != 1 raises ValueError."""
    padder = TemporalPad1d(padding=2, in_channels=1, buffer=None, causal=True)
    x = mx.ones((2, 4, 1))
    
    with pytest.raises(ValueError, match="batch size"):
        padder(x, inference=True)


def test_temporal_pad1d_inference_multiple_chunks():
    """Test streaming inference across multiple chunks."""
    pad_len = 2
    in_channels = 1
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)

    # First chunk
    x1 = mx.array([[[1.0], [2.0]]])
    padded1 = padder(x1, inference=True)
    
    assert padded1.shape == (1, 4, in_channels)
    padded1_flat = mx.reshape(padded1, (-1,))
    assert_array_equal(padded1_flat, [0.0, 0.0, 1.0, 2.0])
    
    # Second chunk
    x2 = mx.array([[[3.0], [4.0]]])
    padded2 = padder(x2, inference=True)
    
    assert padded2.shape == (1, 4, in_channels)
    padded2_flat = mx.reshape(padded2, (-1,))
    assert_array_equal(padded2_flat, [1.0, 2.0, 3.0, 4.0])
    
    # Third chunk
    x3 = mx.array([[[5.0], [6.0]]])
    padded3 = padder(x3, inference=True)
    
    padded3_flat = mx.reshape(padded3, (-1,))
    assert_array_equal(padded3_flat, [3.0, 4.0, 5.0, 6.0])


def test_temporal_pad1d_inference_with_buffer_io():
    """Test streaming inference with BufferIO for multi-layer pipeline."""
    pad_len = 3
    in_channels = 1
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)

    bio = BufferIO()
    
    # First pass - BufferIO is empty, should use internal buffer
    x1 = mx.array([[[1.0], [2.0], [3.0]]])
    padded1 = padder(x1, inference=True, buffer_io=bio)
    
    assert padded1.shape == (1, 6, in_channels)
    assert len(bio.out_buffers) == 1
    assert len(bio.internal_buffer) == 1
    
    # Step to move buffers
    bio.step()
    
    # Second pass
    x2 = mx.array([[[4.0], [5.0], [6.0]]])
    padded2 = padder(x2, inference=True, buffer_io=bio)
    
    padded2_flat = mx.reshape(padded2, (-1,))
    # Should have context from previous chunk
    assert padded2_flat[0].item() == 1.0  # From previous chunk


def test_temporal_pad1d_inference_with_buffer_io_none_returns_internal():
    """Test that when BufferIO returns None, internal buffer is used."""
    pad_len = 2
    in_channels = 1
    initial_buffer = mx.ones((1, pad_len, in_channels)) * 9.0
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=initial_buffer, causal=True)

    bio = BufferIO()  # Empty BufferIO
    
    x = mx.array([[[1.0], [2.0]]])
    padded = padder(x, inference=True, buffer_io=bio)
    
    padded_flat = mx.reshape(padded, (-1,))
    # First two values should be from initial_buffer (9.0)
    assert_array_equal(padded_flat[:2], [9.0, 9.0])


# ============================================================================
# Buffer Management Tests
# ============================================================================

def test_temporal_pad1d_reset_buffer_zeros_contents():
    """Test reset_buffer() zeros out the buffer."""
    padder = TemporalPad1d(padding=2, in_channels=1, buffer=1.0, causal=True)

    padder.reset_buffer()

    assert_array_equal(padder.buffer, mx.zeros_like(padder.buffer))


def test_temporal_pad1d_reset_buffer_after_inference():
    """Test reset_buffer() can be called after inference to clear state."""
    pad_len = 2
    padder = TemporalPad1d(padding=pad_len, in_channels=1, buffer=None, causal=True)

    # Run inference to update buffer
    x = mx.array([[[5.0], [6.0]]])
    padder(x, inference=True)
    
    # Buffer should now contain [5.0, 6.0]
    assert not mx.array_equal(padder.buffer, mx.zeros_like(padder.buffer))
    
    # Reset
    padder.reset_buffer()
    
    # Buffer should be zeros
    assert_array_equal(padder.buffer, 0.0)


def test_temporal_pad1d_buffer_shape_validation():
    """Test that reset_buffer validates buffer length."""
    padder = TemporalPad1d(padding=3, in_channels=1, buffer=None, causal=True)
    
    # Manually corrupt buffer shape
    padder.buffer = mx.zeros((1, 5, 1))  # Wrong length
    
    with pytest.raises(ValueError, match="Buffer length must be"):
        padder.reset_buffer()


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

def test_temporal_pad1d_single_timestep_inference():
    """Test streaming inference with single timestep chunks."""
    pad_len = 3
    padder = TemporalPad1d(padding=pad_len, in_channels=1, buffer=None, causal=True)

    x = mx.array([[[1.0]]])
    padded = padder(x, inference=True)
    
    assert padded.shape == (1, 4, 1)
    padded_flat = mx.reshape(padded, (-1,))
    assert_array_equal(padded_flat, [0.0, 0.0, 0.0, 1.0])


def test_temporal_pad1d_large_sequence_offline():
    """Test offline padding with large sequence."""
    pad_len = 10
    seq_len = 100
    in_channels = 8
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=False)

    x = mx.random.normal((1, seq_len, in_channels))
    padded = padder(x, inference=False)

    assert padded.shape == (1, seq_len + pad_len, in_channels)


def test_temporal_pad1d_preserves_channel_values():
    """Test that padding preserves channel independence."""
    pad_len = 2
    in_channels = 3
    padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None, causal=True)

    x = mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    padded = padder(x, inference=False)

    # Check that original values are preserved
    assert_array_equal(padded[:, pad_len:, :], x)


def test_temporal_pad1d_all_padding_modes():
    """Test that all defined padding modes work."""
    pad_len = 2
    in_channels = 1
    x = mx.array([[[1.0], [2.0], [3.0], [4.0]]])

    for mode in PADDING_MODES:
        padder = TemporalPad1d(padding=pad_len, in_channels=in_channels, buffer=None,
                              padding_mode=mode, causal=False)
        padded = padder(x, inference=False)
        assert padded.shape == (1, 6, in_channels), f"Failed for mode: {mode}"


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # Initialization tests
        test_temporal_pad1d_init_with_none_buffer,
        test_temporal_pad1d_init_with_float_buffer,
        test_temporal_pad1d_init_with_array_buffer,
        test_temporal_pad1d_init_invalid_buffer_type,
        test_temporal_pad1d_init_invalid_padding_mode,
        test_temporal_pad1d_init_causal_left_padding,
        test_temporal_pad1d_init_non_causal_symmetric_padding,
        test_temporal_pad1d_init_non_causal_odd_padding,
        # Offline padding tests
        test_temporal_pad1d_offline_zero_padding,
        test_temporal_pad1d_offline_causal_zero_padding,
        test_temporal_pad1d_offline_replicate_padding,
        test_temporal_pad1d_offline_batch_size_greater_than_one,
        # Inference tests
        test_temporal_pad1d_causal_inference_updates_buffer,
        test_temporal_pad1d_inference_non_causal_raises,
        test_temporal_pad1d_inference_batch_size_not_one_raises,
        test_temporal_pad1d_inference_multiple_chunks,
        test_temporal_pad1d_inference_with_buffer_io,
        test_temporal_pad1d_inference_with_buffer_io_none_returns_internal,
        # Buffer management tests
        test_temporal_pad1d_reset_buffer_zeros_contents,
        test_temporal_pad1d_reset_buffer_after_inference,
        test_temporal_pad1d_buffer_shape_validation,
        # Edge cases
        test_temporal_pad1d_single_timestep_inference,
        test_temporal_pad1d_large_sequence_offline,
        test_temporal_pad1d_preserves_channel_values,
        test_temporal_pad1d_all_padding_modes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print(f"{'='*70}")
    
    if failed > 0:
        sys.exit(1)
