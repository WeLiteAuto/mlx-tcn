import os
import sys

import mlx.core as mx

tests_dir = os.path.dirname(__file__)
project_src = os.path.abspath(os.path.join(tests_dir, "..", "..", "mlx-tcn"))
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from conv import TemporalConv1d, TemporalConvTransposed1d  # noqa: E402
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
    actual_mx = actual if isinstance(actual, mx.array) else mx.array(actual)
    
    if isinstance(expected, (int, float)):
        expected_mx = mx.full(actual_mx.shape, expected)
    else:
        expected_mx = expected if isinstance(expected, mx.array) else mx.array(expected)
    
    assert actual_mx.shape == expected_mx.shape, \
        f"Shape mismatch: {actual_mx.shape} vs {expected_mx.shape}"
    
    assert mx.allclose(actual_mx, expected_mx, rtol=rtol, atol=atol).item(), \
        f"Arrays not equal:\nActual: {actual_mx}\nExpected: {expected_mx}\nDiff: {mx.abs(actual_mx - expected_mx)}"


# ============================================================================
# TemporalConv1d Tests
# ============================================================================

# --- Initialization Tests ---

def test_temporal_conv1d_init_basic():
    """Test basic initialization of TemporalConv1d."""
    in_ch, out_ch, k_size = 8, 16, 3
    conv = TemporalConv1d(in_ch, out_ch, k_size, stride=1)
    
    # Check weight shape to verify channels
    assert 'weight' in conv
    assert conv['weight'].shape == (out_ch, k_size, in_ch)
    assert conv.stride == 1  # MLX stores as int
    assert conv.pad_len == (k_size - 1) * 1  # (3-1)*1 = 2
    assert conv.causal == False


def test_temporal_conv1d_init_with_dilation():
    """Test initialization with dilation."""
    conv = TemporalConv1d(4, 8, kernel_size=3, stride=1, dilation=2)
    
    # MLX stores dilation as integer, not tuple
    assert conv.pad_len == (3 - 1) * 2  # = 4
    assert conv.padder.pad_len == 4


def test_temporal_conv1d_init_causal():
    """Test causal mode initialization."""
    conv = TemporalConv1d(4, 8, kernel_size=5, stride=1, causal=True)
    
    assert conv.causal == True
    assert conv.padder.causal == True
    assert conv.padder.left_padding == conv.pad_len
    assert conv.padder.right_padding == 0


def test_temporal_conv1d_init_with_buffer():
    """Test initialization with custom buffer."""
    in_ch, k_size = 4, 3
    pad_len = (k_size - 1) * 1
    custom_buffer = mx.ones((1, pad_len, in_ch)) * 5.0
    
    conv = TemporalConv1d(in_ch, 8, k_size, stride=1, buffer=custom_buffer, causal=True)
    
    assert mx.array_equal(conv.padder.buffer, custom_buffer)


def test_temporal_conv1d_init_with_float_buffer():
    """Test initialization with float buffer value."""
    conv = TemporalConv1d(4, 8, kernel_size=3, stride=1, buffer=2.5, causal=True)
    
    expected_buffer = mx.full((1, 2, 4), 2.5)
    assert_array_equal(conv.padder.buffer, expected_buffer)


def test_temporal_conv1d_init_invalid_padding_arg():
    """Test that non-zero padding argument raises error."""
    with pytest.raises(ValueError, match="padding"):
        TemporalConv1d(4, 8, kernel_size=3, stride=1, padding=1)


def test_temporal_conv1d_init_invalid_look_ahead():
    """Test that non-zero look_ahead raises error."""
    with pytest.raises(ValueError, match="look_ahead"):
        TemporalConv1d(4, 8, kernel_size=3, stride=1, look_ahead=1)


def test_temporal_conv1d_init_invalid_buffer_shape():
    """Test that invalid buffer shape raises error."""
    wrong_buffer = mx.ones((1, 5, 4))  # Wrong shape
    
    with pytest.raises(ValueError, match="buffer must have shape"):
        TemporalConv1d(4, 8, kernel_size=3, stride=1, buffer=wrong_buffer, causal=True)


def test_temporal_conv1d_init_invalid_buffer_type():
    """Test that invalid buffer type raises error."""
    with pytest.raises(TypeError, match="buffer must be"):
        TemporalConv1d(4, 8, kernel_size=3, stride=1, buffer="invalid", causal=True)


# --- Forward Pass Tests ---

def test_temporal_conv1d_forward_offline():
    """Test offline forward pass (training mode)."""
    batch, seq_len, in_ch = 2, 10, 4
    conv = TemporalConv1d(in_ch, 8, kernel_size=3, stride=1)
    
    x = mx.random.normal((batch, seq_len, in_ch))
    out = conv(x, inference=False)
    
    # Output sequence length should be same as input (due to padding)
    assert out.shape == (batch, seq_len, 8)


def test_temporal_conv1d_forward_with_stride():
    """Test forward pass with stride > 1."""
    batch, seq_len, in_ch = 1, 12, 4
    conv = TemporalConv1d(in_ch, 8, kernel_size=3, stride=2)
    
    x = mx.random.normal((batch, seq_len, in_ch))
    out = conv(x, inference=False)
    
    # With stride=2, output is downsampled
    # Just check batch and channels, length depends on padding
    assert out.shape[0] == batch
    assert out.shape[-1] == 8
    assert out.shape[1] < seq_len  # Should be downsampled


def test_temporal_conv1d_forward_causal_inference():
    """Test causal inference mode."""
    in_ch, out_ch = 2, 4
    conv = TemporalConv1d(in_ch, out_ch, kernel_size=3, stride=1, causal=True)
    
    x = mx.random.normal((1, 5, in_ch))
    out = conv(x, inference=True)
    
    # With causal padding, output length = input + pad_len
    # But conv reduces it back
    assert out.shape[0] == 1
    assert out.shape[-1] == out_ch


def test_temporal_conv1d_streaming_multiple_chunks():
    """Test streaming inference across multiple chunks."""
    in_ch, out_ch = 2, 4
    conv = TemporalConv1d(in_ch, out_ch, kernel_size=3, stride=1, causal=True)
    
    # First chunk
    x1 = mx.random.normal((1, 4, in_ch))
    out1 = conv(x1, inference=True)
    
    # Second chunk - buffer should be updated from first chunk
    x2 = mx.random.normal((1, 4, in_ch))
    out2 = conv(x2, inference=True)
    
    # Both outputs should have same shape
    assert out1.shape == out2.shape
    # Buffer should be updated
    assert not mx.allclose(conv.padder.buffer, mx.zeros_like(conv.padder.buffer)).item()


def test_temporal_conv1d_reset_buffer():
    """Test reset_buffer clears streaming state."""
    conv = TemporalConv1d(4, 8, kernel_size=3, stride=1, causal=True)
    
    # Run inference to populate buffer
    x = mx.random.normal((1, 5, 4))
    conv(x, inference=True)
    
    # Reset buffer
    conv.reset_buffer()
    
    # Buffer should be zeros
    assert_array_equal(conv.padder.buffer, 0.0)


def test_temporal_conv1d_with_buffer_io():
    """Test TemporalConv1d with BufferIO for multi-layer streaming."""
    in_ch, out_ch = 4, 8
    conv = TemporalConv1d(in_ch, out_ch, kernel_size=3, stride=1, causal=True)
    
    bio = BufferIO()
    
    # First pass
    x1 = mx.random.normal((1, 5, in_ch))
    out1 = conv(x1, inference=True, buffer_io=bio)
    
    assert len(bio.out_buffers) == 1
    assert len(bio.internal_buffer) == 1


def test_temporal_conv1d_groups():
    """Test grouped convolution."""
    in_ch, out_ch = 8, 16
    conv = TemporalConv1d(in_ch, out_ch, kernel_size=3, stride=1, groups=2)
    
    x = mx.random.normal((1, 10, in_ch))
    out = conv(x, inference=False)
    
    assert out.shape == (1, 10, out_ch)


def test_temporal_conv1d_no_bias():
    """Test convolution without bias."""
    conv = TemporalConv1d(4, 8, kernel_size=3, stride=1, bias=False)
    
    # Check bias not in parameters
    assert 'bias' not in conv
    
    x = mx.random.normal((1, 10, 4))
    out = conv(x, inference=False)
    
    assert out.shape == (1, 10, 8)


def test_temporal_conv1d_padding_mode_replicate():
    """Test with replicate padding mode."""
    conv = TemporalConv1d(4, 8, kernel_size=3, stride=1, padding_mode="replicate")
    
    assert conv.padder.padding_mode == "replicate"
    
    x = mx.random.normal((1, 10, 4))
    out = conv(x, inference=False)
    
    assert out.shape == (1, 10, 8)


# --- Output Shape Tests ---

def test_temporal_conv1d_output_shapes_various_configs():
    """Test output shapes with various kernel sizes and dilations."""
    configs = [
        # (kernel_size, dilation, stride, seq_len)
        (3, 1, 1, 10),
        (5, 1, 1, 12),
        (3, 2, 1, 10),
        (7, 1, 2, 16),
    ]
    
    for k, d, s, seq_len in configs:
        conv = TemporalConv1d(4, 8, kernel_size=k, stride=s, dilation=d)
        x = mx.random.normal((1, seq_len, 4))
        out = conv(x, inference=False)
        
        # Check output has correct channels
        assert out.shape[-1] == 8, f"Failed for k={k}, d={d}, s={s}"


# ============================================================================
# TemporalConvTransposed1d Tests
# ============================================================================

# --- Initialization Tests ---

def test_temporal_conv_transposed1d_init_basic():
    """Test basic initialization of TemporalConvTransposed1d."""
    in_ch, out_ch = 16, 8
    stride = 2
    kernel_size = 2 * stride  # Required constraint
    
    conv_t = TemporalConvTransposed1d(in_ch, out_ch, kernel_size, stride)
    
    # Check weight exists and upsampling settings
    assert 'weight' in conv_t
    # For transpose conv, weight shape is (in_ch, kernel_size, out_ch)
    assert conv_t.upsampling_factor == stride
    assert conv_t.buffer_size == 1  # (4/2) - 1
    assert conv_t.causal == False


def test_temporal_conv_transposed1d_init_causal():
    """Test causal mode initialization."""
    conv_t = TemporalConvTransposed1d(16, 8, kernel_size=4, stride=2, causal=True)
    
    assert conv_t.causal == True
    assert conv_t.implicit_padding == 0
    assert conv_t.padder.causal == True


def test_temporal_conv_transposed1d_init_non_causal():
    """Test non-causal mode implicit padding."""
    stride = 2
    kernel_size = 4
    conv_t = TemporalConvTransposed1d(16, 8, kernel_size, stride, causal=False)
    
    expected_padding = (kernel_size - stride) // 2  # (4-2)//2 = 1
    assert conv_t.implicit_padding == expected_padding


def test_temporal_conv_transposed1d_invalid_padding():
    """Test that non-zero padding raises error."""
    with pytest.raises(ValueError, match="padding"):
        TemporalConvTransposed1d(16, 8, kernel_size=4, stride=2, padding=1)


def test_temporal_conv_transposed1d_invalid_dilation():
    """Test that non-unity dilation raises error."""
    with pytest.raises(ValueError, match="dilation"):
        TemporalConvTransposed1d(16, 8, kernel_size=4, stride=2, dilation=2)


def test_temporal_conv_transposed1d_invalid_look_ahead():
    """Test that non-zero look_ahead raises error."""
    with pytest.raises(ValueError, match="look_ahead"):
        TemporalConvTransposed1d(16, 8, kernel_size=4, stride=2, look_ahead=1)


def test_temporal_conv_transposed1d_invalid_output_padding():
    """Test that non-zero output_padding raises error."""
    with pytest.raises(ValueError, match="output_padding"):
        TemporalConvTransposed1d(16, 8, kernel_size=4, stride=2, output_padding=1)


def test_temporal_conv_transposed1d_invalid_kernel_stride_ratio():
    """Test that kernel_size != 2*stride raises error."""
    with pytest.raises(ValueError, match="kernel_size == 2"):
        TemporalConvTransposed1d(16, 8, kernel_size=5, stride=2)


# --- Forward Pass Tests ---

def test_temporal_conv_transposed1d_forward_offline():
    """Test offline forward pass."""
    batch, seq_len, in_ch = 1, 10, 16
    stride = 2
    conv_t = TemporalConvTransposed1d(in_ch, 8, kernel_size=4, stride=stride)
    
    x = mx.random.normal((batch, seq_len, in_ch))
    out = conv_t(x, inference=False)
    
    # Output should be upsampled by stride factor
    # For non-causal, output_len ≈ input_len * stride
    assert out.shape[0] == batch
    assert out.shape[-1] == 8
    # Check upsampling occurred
    assert out.shape[1] > seq_len


def test_temporal_conv_transposed1d_upsampling_factor():
    """Test that upsampling works correctly."""
    in_ch, out_ch = 16, 8
    seq_len = 5
    
    for stride in [2, 4]:
        conv_t = TemporalConvTransposed1d(in_ch, out_ch, kernel_size=2*stride, stride=stride)
        
        x = mx.random.normal((1, seq_len, in_ch))
        out = conv_t(x, inference=False)
        
        # Non-causal upsampling
        assert out.shape[-1] == out_ch
        assert out.shape[1] >= seq_len * stride - stride  # Approximate check


def test_temporal_conv_transposed1d_causal_forward():
    """Test causal mode forward pass."""
    in_ch, out_ch = 16, 8
    stride = 2
    conv_t = TemporalConvTransposed1d(in_ch, out_ch, kernel_size=4, stride=stride, causal=True)
    
    x = mx.random.normal((1, 5, in_ch))
    out = conv_t(x, inference=False)
    
    assert out.shape[-1] == out_ch
    # In causal mode, edges are trimmed


def test_temporal_conv_transposed1d_causal_inference():
    """Test causal inference mode with buffering."""
    in_ch, out_ch = 8, 4
    stride = 2
    conv_t = TemporalConvTransposed1d(in_ch, out_ch, kernel_size=4, stride=stride, causal=True)
    
    x = mx.random.normal((1, 5, in_ch))
    out = conv_t(x, inference=True)
    
    assert out.shape[0] == 1
    assert out.shape[-1] == out_ch


def test_temporal_conv_transposed1d_streaming():
    """Test streaming inference across multiple chunks."""
    in_ch, out_ch = 8, 4
    stride = 2
    conv_t = TemporalConvTransposed1d(in_ch, out_ch, kernel_size=4, stride=stride, causal=True)
    
    # First chunk
    x1 = mx.random.normal((1, 4, in_ch))
    out1 = conv_t(x1, inference=True)
    
    # Second chunk
    x2 = mx.random.normal((1, 4, in_ch))
    out2 = conv_t(x2, inference=True)
    
    # Both should have same channel dimension
    assert out1.shape[-1] == out2.shape[-1] == out_ch


def test_temporal_conv_transposed1d_reset_buffer():
    """Test reset_buffer clears streaming state."""
    conv_t = TemporalConvTransposed1d(8, 4, kernel_size=4, stride=2, causal=True)
    
    # Run inference to populate buffer
    x = mx.random.normal((1, 5, 8))
    conv_t(x, inference=True)
    
    # Reset buffer
    conv_t.reset_buffer()
    
    # Buffer should be zeros
    assert_array_equal(conv_t.padder.buffer, 0.0)


def test_temporal_conv_transposed1d_with_buffer_io():
    """Test with BufferIO for multi-layer streaming."""
    conv_t = TemporalConvTransposed1d(8, 4, kernel_size=4, stride=2, causal=True)
    
    bio = BufferIO()
    
    x = mx.random.normal((1, 5, 8))
    out = conv_t(x, inference=True, buffer_io=bio)
    
    assert len(bio.out_buffers) == 1
    assert len(bio.internal_buffer) == 1


def test_temporal_conv_transposed1d_odd_stride_non_causal():
    """Test non-causal mode with odd stride."""
    stride = 3
    conv_t = TemporalConvTransposed1d(8, 4, kernel_size=6, stride=stride, causal=False)
    
    x = mx.random.normal((1, 5, 8))
    out = conv_t(x, inference=False)
    
    # For odd stride, last element is trimmed
    assert out.shape[-1] == 4


# --- Integration Tests ---

def test_temporal_conv1d_and_transposed_symmetry():
    """Test that Conv1d followed by ConvTranspose1d approximately recovers shape."""
    in_ch, mid_ch = 4, 8
    seq_len = 10
    stride = 2
    
    # Downsampling
    conv_down = TemporalConv1d(in_ch, mid_ch, kernel_size=3, stride=stride)
    
    # Upsampling
    conv_up = TemporalConvTransposed1d(mid_ch, in_ch, kernel_size=4, stride=stride)
    
    x = mx.random.normal((1, seq_len, in_ch))
    
    # Forward pass
    down = conv_down(x, inference=False)
    up = conv_up(down, inference=False)
    
    # Shape should be approximately recovered
    assert up.shape[-1] == in_ch
    # Length might differ slightly due to padding/trimming
    assert abs(up.shape[1] - seq_len) <= stride


def test_encoder_decoder_pipeline():
    """Test a simple encoder-decoder pipeline."""
    in_ch = 4
    hidden_ch = 8
    
    # Encoder
    encoder = TemporalConv1d(in_ch, hidden_ch, kernel_size=3, stride=2, causal=True)
    
    # Decoder
    decoder = TemporalConvTransposed1d(hidden_ch, in_ch, kernel_size=4, stride=2, causal=True)
    
    x = mx.random.normal((1, 12, in_ch))
    
    # Encode
    encoded = encoder(x, inference=True)
    
    # Decode
    decoded = decoder(encoded, inference=True)
    
    assert decoded.shape[-1] == in_ch


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # TemporalConv1d initialization tests
        test_temporal_conv1d_init_basic,
        test_temporal_conv1d_init_with_dilation,
        test_temporal_conv1d_init_causal,
        test_temporal_conv1d_init_with_buffer,
        test_temporal_conv1d_init_with_float_buffer,
        test_temporal_conv1d_init_invalid_padding_arg,
        test_temporal_conv1d_init_invalid_look_ahead,
        test_temporal_conv1d_init_invalid_buffer_shape,
        test_temporal_conv1d_init_invalid_buffer_type,
        
        # TemporalConv1d forward pass tests
        test_temporal_conv1d_forward_offline,
        test_temporal_conv1d_forward_with_stride,
        test_temporal_conv1d_forward_causal_inference,
        test_temporal_conv1d_streaming_multiple_chunks,
        test_temporal_conv1d_reset_buffer,
        test_temporal_conv1d_with_buffer_io,
        test_temporal_conv1d_groups,
        test_temporal_conv1d_no_bias,
        test_temporal_conv1d_padding_mode_replicate,
        test_temporal_conv1d_output_shapes_various_configs,
        
        # TemporalConvTransposed1d initialization tests
        test_temporal_conv_transposed1d_init_basic,
        test_temporal_conv_transposed1d_init_causal,
        test_temporal_conv_transposed1d_init_non_causal,
        test_temporal_conv_transposed1d_invalid_padding,
        test_temporal_conv_transposed1d_invalid_dilation,
        test_temporal_conv_transposed1d_invalid_look_ahead,
        test_temporal_conv_transposed1d_invalid_output_padding,
        test_temporal_conv_transposed1d_invalid_kernel_stride_ratio,
        
        # TemporalConvTransposed1d forward pass tests
        test_temporal_conv_transposed1d_forward_offline,
        test_temporal_conv_transposed1d_upsampling_factor,
        test_temporal_conv_transposed1d_causal_forward,
        test_temporal_conv_transposed1d_causal_inference,
        test_temporal_conv_transposed1d_streaming,
        test_temporal_conv_transposed1d_reset_buffer,
        test_temporal_conv_transposed1d_with_buffer_io,
        test_temporal_conv_transposed1d_odd_stride_non_causal,
        
        # Integration tests
        test_temporal_conv1d_and_transposed_symmetry,
        test_encoder_decoder_pipeline,
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

