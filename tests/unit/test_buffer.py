import os
import sys

import mlx.core as mx

tests_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(tests_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mlx_tcn import BufferIO  # noqa: E402


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


# Replace pytest.raises with our simple version
class pytest:
    raises = staticmethod(pytest_raises)


def test_buffer_io_init_empty():
    """Test BufferIO initialization with no input buffers."""
    bio = BufferIO()
    
    assert bio.in_buffer is None
    assert bio.in_buffer_length == 0
    assert bio.out_buffers == []
    assert bio.internal_buffer == []


def test_buffer_io_init_with_buffers():
    """Test BufferIO initialization with precomputed input buffers."""
    input_buffers = [mx.array([1, 2, 3]), mx.array([4, 5, 6]), mx.array([7, 8, 9])]
    bio = BufferIO(in_buffers=input_buffers)
    
    assert bio.in_buffer is not None
    assert bio.in_buffer_length == 3
    assert bio.out_buffers == []
    assert bio.internal_buffer == []


def test_buffer_io_iterator():
    """Test that BufferIO can be used as an iterator with None termination."""
    input_buffers = [mx.array([1]), mx.array([2]), mx.array([3])]
    bio = BufferIO(in_buffers=input_buffers)
    
    results = []
    for buf in bio:
        if buf is None:
            break
        results.append(buf)
    
    assert len(results) == 3
    assert mx.array_equal(results[0], mx.array([1]))
    assert mx.array_equal(results[1], mx.array([2]))
    assert mx.array_equal(results[2], mx.array([3]))


def test_buffer_io_next():
    """Test __next__ method for accessing input buffers."""
    input_buffers = [mx.array([10, 20]), mx.array([30, 40])]
    bio = BufferIO(in_buffers=input_buffers)
    
    buf1 = next(bio)
    assert mx.array_equal(buf1, mx.array([10, 20]))
    
    buf2 = next(bio)
    assert mx.array_equal(buf2, mx.array([30, 40]))
    
    # Should return None when exhausted
    buf3 = next(bio)
    assert buf3 is None


def test_buffer_io_next_empty():
    """Test __next__ on empty BufferIO returns None."""
    bio = BufferIO()
    
    # Should return None when in_buffer is None
    result = next(bio)
    assert result is None


def test_buffer_io_next_in_buffer():
    """Test next_in_buffer method returns buffer or None."""
    input_buffers = [mx.array([1, 2]), mx.array([3, 4])]
    bio = BufferIO(in_buffers=input_buffers)
    
    buf1 = bio.next_in_buffer()
    assert buf1 is not None
    assert mx.array_equal(buf1, mx.array([1, 2]))
    
    buf2 = bio.next_in_buffer()
    assert buf2 is not None
    assert mx.array_equal(buf2, mx.array([3, 4]))
    
    # next_in_buffer should return None when exhausted instead of raising
    buf3 = bio.next_in_buffer()
    assert buf3 is None


def test_buffer_io_next_in_buffer_empty():
    """Test next_in_buffer on empty BufferIO returns None."""
    bio = BufferIO()
    
    # Should return None when in_buffer is None
    result = bio.next_in_buffer()
    assert result is None


def test_buffer_io_append_out_buffer():
    """Test appending output buffers."""
    bio = BufferIO()
    
    bio.append_out_buffer(mx.array([1, 2, 3]))
    bio.append_out_buffer(mx.array([4, 5, 6]))
    
    assert len(bio.out_buffers) == 2
    assert mx.array_equal(bio.out_buffers[0], mx.array([1, 2, 3]))
    assert mx.array_equal(bio.out_buffers[1], mx.array([4, 5, 6]))


def test_buffer_io_append_internal_buffer():
    """Test appending internal buffers."""
    bio = BufferIO()
    
    bio.append_internal_buffer(mx.array([10, 20]))
    bio.append_internal_buffer(mx.array([30, 40]))
    
    assert len(bio.internal_buffer) == 2
    assert mx.array_equal(bio.internal_buffer[0], mx.array([10, 20]))
    assert mx.array_equal(bio.internal_buffer[1], mx.array([30, 40]))


def test_buffer_io_step_basic():
    """Test step() moves out_buffers to in_buffer."""
    input_buffers = [mx.array([1]), mx.array([2])]
    bio = BufferIO(in_buffers=input_buffers)
    
    # Consume input buffers
    next(bio)
    next(bio)
    
    # Produce output buffers
    bio.append_out_buffer(mx.array([10]))
    bio.append_out_buffer(mx.array([20]))
    
    # Step should move out_buffers to in_buffer
    bio.step()
    
    assert len(bio.out_buffers) == 0
    assert bio.in_buffer_length == 2
    
    # Should be able to iterate over new buffers
    buf1 = next(bio)
    assert mx.array_equal(buf1, mx.array([10]))
    
    buf2 = next(bio)
    assert mx.array_equal(buf2, mx.array([20]))


def test_buffer_io_step_with_internal_buffer():
    """Test step() uses internal_buffer when in_buffer is None."""
    bio = BufferIO()
    
    # Add internal buffers
    bio.append_internal_buffer(mx.array([5]))
    bio.append_internal_buffer(mx.array([6]))
    
    # First step should use internal_buffer
    bio.append_out_buffer(mx.array([50]))
    bio.append_out_buffer(mx.array([60]))
    bio.step()
    
    assert len(bio.internal_buffer) == 0
    assert bio.in_buffer_length == 2
    
    # Should iterate over out_buffers that became in_buffer
    buf1 = next(bio)
    assert mx.array_equal(buf1, mx.array([50]))


def test_buffer_io_step_mismatch_raises_error():
    """Test step() raises ValueError when buffer counts don't match."""
    input_buffers = [mx.array([1]), mx.array([2])]
    bio = BufferIO(in_buffers=input_buffers)
    
    # Consume input buffers
    next(bio)
    next(bio)
    
    # Only produce 1 output buffer (mismatch)
    bio.append_out_buffer(mx.array([10]))
    
    with pytest.raises(ValueError, match="number of out buffers must be equal"):
        bio.step()


def test_buffer_io_multiple_steps():
    """Test multiple step() calls in sequence."""
    input_buffers = [mx.array([1]), mx.array([2])]
    bio = BufferIO(in_buffers=input_buffers)
    
    # Step 1
    buf1 = next(bio)
    buf2 = next(bio)
    bio.append_out_buffer(mx.array([10]))
    bio.append_out_buffer(mx.array([20]))
    bio.step()
    
    # Step 2
    buf3 = next(bio)
    buf4 = next(bio)
    assert mx.array_equal(buf3, mx.array([10]))
    assert mx.array_equal(buf4, mx.array([20]))
    
    bio.append_out_buffer(mx.array([100]))
    bio.append_out_buffer(mx.array([200]))
    bio.step()
    
    # Step 3
    buf5 = next(bio)
    buf6 = next(bio)
    assert mx.array_equal(buf5, mx.array([100]))
    assert mx.array_equal(buf6, mx.array([200]))


def test_buffer_io_streaming_workflow():
    """Test realistic streaming workflow with BufferIO."""
    # Simulate a 3-layer streaming model
    bio = BufferIO()
    
    # Initialize internal buffers for 3 layers
    bio.append_internal_buffer(mx.zeros((1, 4, 8)))
    bio.append_internal_buffer(mx.zeros((1, 4, 16)))
    bio.append_internal_buffer(mx.zeros((1, 4, 32)))
    
    # Need to call step() first to move internal_buffer to in_buffer
    # Since there are no out_buffers yet, we need to handle this differently
    # Let's simulate the first step manually
    bio.in_buffer_length = len(bio.internal_buffer)
    bio.in_buffer = iter(bio.internal_buffer)
    bio.internal_buffer = []
    
    # Process first chunk
    i = 0
    for layer_buffer in bio:
        if layer_buffer is None:
            break
        # Simulate processing with each layer's buffer
        processed = layer_buffer + mx.ones_like(layer_buffer) * (i + 1)
        bio.append_out_buffer(processed)
        i += 1
    
    bio.step()
    
    # Process second chunk - buffers should have updated values
    i = 0
    for layer_buffer in bio:
        if layer_buffer is None:
            break
        assert layer_buffer.shape == [(1, 4, 8), (1, 4, 16), (1, 4, 32)][i]
        # First layer buffer should be all 1s, second all 2s, third all 3s
        expected = mx.ones_like(layer_buffer) * (i + 1)
        assert mx.allclose(layer_buffer, expected)
        i += 1


if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        test_buffer_io_init_empty,
        test_buffer_io_init_with_buffers,
        test_buffer_io_iterator,
        test_buffer_io_next,
        test_buffer_io_next_empty,
        test_buffer_io_next_in_buffer,
        test_buffer_io_next_in_buffer_empty,
        test_buffer_io_append_out_buffer,
        test_buffer_io_append_internal_buffer,
        test_buffer_io_step_basic,
        test_buffer_io_step_with_internal_buffer,
        test_buffer_io_step_mismatch_raises_error,
        test_buffer_io_multiple_steps,
        test_buffer_io_streaming_workflow,
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
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)
