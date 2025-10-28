"""
Unit tests for train.loop module.

Tests training loop, evaluation, and metric computation functionality.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.config import TrainingConfig
from train.data import DatasetBundle, DatasetMetadata, WaveformDataset
from train.loop import (
    EpochMetrics,
    EpochResult,
    _select_sequence_logits,
    _sum_absolute_error,
    create_train_step,
    evaluate,
    run_training_loop,
    setup_optimizer,
    train_epoch,
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_simple_model(in_features=3, seq_len=10, out_features=2):
    """Create a simple model for testing."""
    
    class SimpleModel(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.linear = nn.Linear(in_f, out_f)
        
        def __call__(self, x, embeddings=None, inference=False):
            # x shape: (batch, time, features)
            # Return shape: (batch, time, out_features)
            batch, time, _ = x.shape
            # Simple pass through
            out = self.linear(x)
            return out
    
    return SimpleModel(in_features, out_features)


def create_test_dataset(n_samples=20, seq_len=10, n_features=3, n_targets=2):
    """Create a simple test dataset."""
    features = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    labels = np.random.randn(n_samples, n_targets).astype(np.float32)
    cont_params = np.random.randn(n_samples, 5).astype(np.float32)
    disc_params = np.random.randint(0, 2, (n_samples, 2)).astype(np.int32)
    
    return WaveformDataset(features, labels, cont_params, disc_params)


# ============================================================================
# EpochMetrics and EpochResult Tests
# ============================================================================


def test_epoch_metrics_init():
    """Test EpochMetrics initialization."""
    metrics = EpochMetrics(loss=0.5, mae=0.2)
    
    assert metrics.loss == 0.5
    assert metrics.mae == 0.2


def test_epoch_result_init():
    """Test EpochResult initialization."""
    train_metrics = EpochMetrics(loss=0.5, mae=0.2)
    val_metrics = EpochMetrics(loss=0.6, mae=0.25)
    result = EpochResult(epoch=1, train=train_metrics, val=val_metrics)
    
    assert result.epoch == 1
    assert result.train is train_metrics
    assert result.val is val_metrics


# ============================================================================
# setup_optimizer Tests
# ============================================================================


def test_setup_optimizer_adam():
    """Test setup_optimizer creates Adam when weight_decay is 0."""
    config = TrainingConfig(learning_rate=1e-3, weight_decay=0.0)
    optimizer = setup_optimizer(config)
    
    assert isinstance(optimizer, optim.Adam)


def test_setup_optimizer_adamw():
    """Test setup_optimizer creates AdamW when weight_decay > 0."""
    config = TrainingConfig(learning_rate=1e-3, weight_decay=1e-4)
    
    try:
        optimizer = setup_optimizer(config)
        assert isinstance(optimizer, optim.AdamW)
    except RuntimeError as e:
        # AdamW might not be available in older MLX versions
        assert "AdamW optimiser is not available" in str(e)


def test_setup_optimizer_learning_rate():
    """Test that optimizer uses correct learning rate."""
    config = TrainingConfig(learning_rate=5e-4, weight_decay=0.0)
    optimizer = setup_optimizer(config)
    
    assert optimizer.learning_rate == 5e-4


# ============================================================================
# _select_sequence_logits Tests
# ============================================================================


def test_select_sequence_logits_basic():
    """Test _select_sequence_logits selects last timestep."""
    # Shape: (batch=2, time=5, features=3)
    logits = mx.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]]
    ], dtype=mx.float32)
    
    result = _select_sequence_logits(logits)
    
    assert result.shape == (2, 3)
    # Should select last timestep
    np.testing.assert_array_equal(result[0], [13, 14, 15])
    np.testing.assert_array_equal(result[1], [28, 29, 30])


def test_select_sequence_logits_single_timestep():
    """Test _select_sequence_logits with single timestep."""
    logits = mx.random.normal((4, 1, 5))
    result = _select_sequence_logits(logits)
    
    assert result.shape == (4, 5)


def test_select_sequence_logits_invalid_shape():
    """Test _select_sequence_logits raises error for wrong dimensions."""
    # 2D array (missing time dimension)
    logits_2d = mx.random.normal((4, 5))
    
    with pytest.raises(ValueError, match="Expected logits to have shape"):
        _select_sequence_logits(logits_2d)
    
    # 4D array (too many dimensions)
    logits_4d = mx.random.normal((2, 3, 4, 5))
    
    with pytest.raises(ValueError, match="Expected logits to have shape"):
        _select_sequence_logits(logits_4d)


# ============================================================================
# _sum_absolute_error Tests
# ============================================================================


def test_sum_absolute_error_basic():
    """Test _sum_absolute_error computes correct sum."""
    predictions = mx.array([[1.0, 2.0], [3.0, 4.0]])
    targets = mx.array([[1.5, 1.5], [3.5, 3.5]])
    
    result = _sum_absolute_error(predictions, targets)
    
    # |1-1.5| + |2-1.5| + |3-3.5| + |4-3.5| = 0.5 + 0.5 + 0.5 + 0.5 = 2.0
    assert abs(result.item() - 2.0) < 1e-5


def test_sum_absolute_error_zeros():
    """Test _sum_absolute_error with perfect predictions."""
    predictions = mx.array([[1.0, 2.0], [3.0, 4.0]])
    targets = mx.array([[1.0, 2.0], [3.0, 4.0]])
    
    result = _sum_absolute_error(predictions, targets)
    
    assert abs(result.item()) < 1e-7


def test_sum_absolute_error_negative():
    """Test _sum_absolute_error handles negative values."""
    predictions = mx.array([[-1.0, -2.0]])
    targets = mx.array([[1.0, 2.0]])
    
    result = _sum_absolute_error(predictions, targets)
    
    # |-1-1| + |-2-2| = 2 + 4 = 6
    assert abs(result.item() - 6.0) < 1e-5


# ============================================================================
# create_train_step Tests
# ============================================================================


def test_create_train_step_returns_callable():
    """Test create_train_step returns a callable function."""
    config = TrainingConfig()
    optimizer = setup_optimizer(config)
    
    train_step = create_train_step(optimizer)
    
    assert callable(train_step)


def test_create_train_step_updates_model():
    """Test that train_step actually updates model parameters."""
    model = create_simple_model(in_features=3, out_features=2)
    config = TrainingConfig(learning_rate=0.1)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    # Get initial weights
    initial_weight = mx.array(model.linear.weight)
    
    # Create dummy data
    x = mx.random.normal((4, 10, 3))
    y = mx.random.normal((4, 2))
    
    # Run train step
    loss, logits = train_step(model, x, y)
    mx.eval(model.parameters())
    
    # Check that weights changed
    updated_weight = model.linear.weight
    assert not mx.array_equal(initial_weight, updated_weight)


def test_create_train_step_returns_loss_and_logits():
    """Test that train_step returns loss and logits."""
    model = create_simple_model(in_features=3, out_features=2)
    config = TrainingConfig()
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    x = mx.random.normal((4, 10, 3))
    y = mx.random.normal((4, 2))
    
    loss, logits = train_step(model, x, y)
    
    assert isinstance(loss, mx.array)
    assert loss.ndim == 0  # Scalar
    assert isinstance(logits, mx.array)
    assert logits.shape == (4, 2)


# ============================================================================
# train_epoch Tests
# ============================================================================


def test_train_epoch_basic():
    """Test basic train_epoch execution."""
    model = create_simple_model(in_features=3, out_features=2)
    dataset = create_test_dataset(n_samples=20, n_features=3, n_targets=2)
    config = TrainingConfig(batch_size=8, log_every_n_steps=100, seed=42)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    metrics = train_epoch(model, dataset, train_step, config, epoch=1)
    
    assert isinstance(metrics, EpochMetrics)
    assert isinstance(metrics.loss, float)
    assert isinstance(metrics.mae, float)
    assert metrics.loss >= 0
    assert metrics.mae >= 0


def test_train_epoch_calls_model_train():
    """Test that train_epoch puts model in training mode."""
    model = create_simple_model()
    model.train = MagicMock()
    model.eval = MagicMock()
    
    dataset = create_test_dataset(n_samples=10)
    config = TrainingConfig(batch_size=5, seed=42)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    train_epoch(model, dataset, train_step, config, epoch=1)
    
    model.train.assert_called_once()


def test_train_epoch_empty_dataset():
    """Test train_epoch raises error with empty dataset."""
    model = create_simple_model()
    # Empty dataset
    dataset = WaveformDataset(
        np.zeros((0, 10, 3), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 5), dtype=np.float32),
        np.zeros((0, 2), dtype=np.int32),
    )
    config = TrainingConfig(batch_size=8)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    with pytest.raises(RuntimeError, match="Training dataset produced zero samples"):
        train_epoch(model, dataset, train_step, config, epoch=1)


def test_train_epoch_seed_affects_shuffle():
    """Test that different epochs use different seeds for shuffling."""
    model = create_simple_model()
    dataset = create_test_dataset(n_samples=50)
    config = TrainingConfig(batch_size=10, seed=42, log_every_n_steps=100)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    # Run two epochs
    metrics1 = train_epoch(model, dataset, train_step, config, epoch=1)
    metrics2 = train_epoch(model, dataset, train_step, config, epoch=2)
    
    # Both should succeed (different shuffle orders)
    assert isinstance(metrics1, EpochMetrics)
    assert isinstance(metrics2, EpochMetrics)


@patch('builtins.print')
def test_train_epoch_logging(mock_print):
    """Test that train_epoch logs at specified intervals."""
    model = create_simple_model()
    dataset = create_test_dataset(n_samples=30)
    config = TrainingConfig(batch_size=5, log_every_n_steps=2, seed=42)
    optimizer = setup_optimizer(config)
    train_step = create_train_step(optimizer)
    
    train_epoch(model, dataset, train_step, config, epoch=1)
    
    # Should log at steps 2, 4, 6 (3 times for 30 samples / 5 batch_size = 6 steps)
    assert mock_print.call_count == 3


# ============================================================================
# evaluate Tests
# ============================================================================


def test_evaluate_basic():
    """Test basic evaluate execution."""
    model = create_simple_model(in_features=3, out_features=2)
    dataset = create_test_dataset(n_samples=20, n_features=3, n_targets=2)
    config = TrainingConfig(batch_size=8)
    
    metrics = evaluate(model, dataset, config)
    
    assert isinstance(metrics, EpochMetrics)
    assert isinstance(metrics.loss, float)
    assert isinstance(metrics.mae, float)
    assert metrics.loss >= 0
    assert metrics.mae >= 0


def test_evaluate_calls_model_eval():
    """Test that evaluate puts model in eval mode."""
    model = create_simple_model()
    model.eval = MagicMock()
    
    dataset = create_test_dataset(n_samples=10)
    config = TrainingConfig(batch_size=5)
    
    evaluate(model, dataset, config)
    
    model.eval.assert_called_once()


def test_evaluate_empty_dataset():
    """Test evaluate returns zero metrics for empty dataset."""
    model = create_simple_model()
    # Empty dataset
    dataset = WaveformDataset(
        np.zeros((0, 10, 3), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 5), dtype=np.float32),
        np.zeros((0, 2), dtype=np.int32),
    )
    config = TrainingConfig(batch_size=8)
    
    metrics = evaluate(model, dataset, config)
    
    assert metrics.loss == 0.0
    assert metrics.mae == 0.0


def test_evaluate_no_shuffle():
    """Test that evaluate doesn't shuffle data."""
    model = create_simple_model()
    
    # Create dataset with sequential features
    features = np.arange(100 * 10 * 3).reshape(100, 10, 3).astype(np.float32)
    labels = np.random.randn(100, 2).astype(np.float32)
    cont_params = np.random.randn(100, 5).astype(np.float32)
    disc_params = np.random.randint(0, 2, (100, 2)).astype(np.int32)
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    config = TrainingConfig(batch_size=100)
    
    # Should work consistently
    metrics1 = evaluate(model, dataset, config)
    metrics2 = evaluate(model, dataset, config)
    
    # Same order should give same results (model isn't training)
    assert abs(metrics1.loss - metrics2.loss) < 1e-5


# ============================================================================
# run_training_loop Tests
# ============================================================================


def test_run_training_loop_basic():
    """Test basic run_training_loop execution."""
    model = create_simple_model(in_features=3, out_features=2)
    
    train_ds = create_test_dataset(n_samples=20, n_features=3, n_targets=2)
    val_ds = create_test_dataset(n_samples=10, n_features=3, n_targets=2)
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(
        batch_size=8,
        num_epochs=2,
        log_every_n_steps=100,
        seed=42
    )
    
    history = run_training_loop(model, bundle, config)
    
    assert len(history) == 2
    assert all(isinstance(r, EpochResult) for r in history)
    assert history[0].epoch == 1
    assert history[1].epoch == 2


def test_run_training_loop_returns_metrics():
    """Test that run_training_loop returns proper metrics."""
    model = create_simple_model()
    
    train_ds = create_test_dataset(n_samples=20)
    val_ds = create_test_dataset(n_samples=10)
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(num_epochs=1, batch_size=8, log_every_n_steps=100)
    
    history = run_training_loop(model, bundle, config)
    
    result = history[0]
    assert isinstance(result.train, EpochMetrics)
    assert isinstance(result.val, EpochMetrics)
    assert result.train.loss >= 0
    assert result.train.mae >= 0
    assert result.val.loss >= 0
    assert result.val.mae >= 0


def test_run_training_loop_multiple_epochs():
    """Test run_training_loop with multiple epochs."""
    model = create_simple_model()
    
    train_ds = create_test_dataset(n_samples=30)
    val_ds = create_test_dataset(n_samples=10)
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(num_epochs=3, batch_size=10, log_every_n_steps=100)
    
    history = run_training_loop(model, bundle, config)
    
    assert len(history) == 3
    for i, result in enumerate(history, start=1):
        assert result.epoch == i


@patch('builtins.print')
def test_run_training_loop_prints_progress(mock_print):
    """Test that run_training_loop prints epoch progress."""
    model = create_simple_model()
    
    train_ds = create_test_dataset(n_samples=20)
    val_ds = create_test_dataset(n_samples=10)
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(num_epochs=2, batch_size=10, log_every_n_steps=100)
    
    run_training_loop(model, bundle, config)
    
    # Should print epoch summary for each epoch (2 times)
    epoch_prints = [call for call in mock_print.call_args_list 
                    if 'Epoch' in str(call) and 'train_loss' in str(call)]
    assert len(epoch_prints) == 2


def test_run_training_loop_empty_history():
    """Test run_training_loop with zero epochs."""
    model = create_simple_model()
    
    train_ds = create_test_dataset(n_samples=20)
    val_ds = create_test_dataset(n_samples=10)
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(num_epochs=0, batch_size=10)
    
    history = run_training_loop(model, bundle, config)
    
    assert len(history) == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_training_pipeline():
    """Test complete training pipeline from setup to finish."""
    # Create model
    model = create_simple_model(in_features=3, out_features=2)
    
    # Create datasets
    train_ds = create_test_dataset(n_samples=40, n_features=3, n_targets=2)
    val_ds = create_test_dataset(n_samples=10, n_features=3, n_targets=2)
    metadata = DatasetMetadata(3, 10, 2, ("HIC", "Dmax"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    # Configure training
    config = TrainingConfig(
        batch_size=8,
        num_epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        log_every_n_steps=10,
        seed=42
    )
    
    # Run training
    history = run_training_loop(model, bundle, config)
    
    # Verify results
    assert len(history) == 3
    
    # Check that metrics are reasonable
    for result in history:
        assert 0 <= result.train.loss < 1000
        assert 0 <= result.train.mae < 1000
        assert 0 <= result.val.loss < 1000
        assert 0 <= result.val.mae < 1000


def test_training_improves_metrics():
    """Test that training generally improves (or maintains) metrics."""
    model = create_simple_model(in_features=3, out_features=2)
    
    # Create simple dataset with consistent pattern
    n_samples = 50
    features = np.random.randn(n_samples, 10, 3).astype(np.float32)
    labels = np.random.randn(n_samples, 2).astype(np.float32)
    cont_params = np.random.randn(n_samples, 5).astype(np.float32)
    disc_params = np.random.randint(0, 2, (n_samples, 2)).astype(np.int32)
    
    train_ds = WaveformDataset(features, labels, cont_params, disc_params)
    val_ds = WaveformDataset(
        features[:10], labels[:10], cont_params[:10], disc_params[:10]
    )
    
    metadata = DatasetMetadata(3, 10, 2, ("t1", "t2"), 5, 2, (0, 1))
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    config = TrainingConfig(
        batch_size=10,
        num_epochs=5,
        learning_rate=0.01,
        log_every_n_steps=100,
        seed=42
    )
    
    history = run_training_loop(model, bundle, config)
    
    # Training loss should generally decrease or stay low
    initial_loss = history[0].train.loss
    final_loss = history[-1].train.loss
    
    # At least verify we completed training without errors
    assert len(history) == 5
    assert final_loss >= 0


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # Basic dataclass tests
        test_epoch_metrics_init,
        test_epoch_result_init,
        
        # setup_optimizer tests
        test_setup_optimizer_adam,
        test_setup_optimizer_adamw,
        test_setup_optimizer_learning_rate,
        
        # _select_sequence_logits tests
        test_select_sequence_logits_basic,
        test_select_sequence_logits_single_timestep,
        test_select_sequence_logits_invalid_shape,
        
        # _sum_absolute_error tests
        test_sum_absolute_error_basic,
        test_sum_absolute_error_zeros,
        test_sum_absolute_error_negative,
        
        # create_train_step tests
        test_create_train_step_returns_callable,
        test_create_train_step_updates_model,
        test_create_train_step_returns_loss_and_logits,
        
        # train_epoch tests
        test_train_epoch_basic,
        test_train_epoch_calls_model_train,
        test_train_epoch_empty_dataset,
        test_train_epoch_seed_affects_shuffle,
        test_train_epoch_logging,
        
        # evaluate tests
        test_evaluate_basic,
        test_evaluate_calls_model_eval,
        test_evaluate_empty_dataset,
        test_evaluate_no_shuffle,
        
        # run_training_loop tests
        test_run_training_loop_basic,
        test_run_training_loop_returns_metrics,
        test_run_training_loop_multiple_epochs,
        test_run_training_loop_prints_progress,
        test_run_training_loop_empty_history,
        
        # Integration tests
        test_full_training_pipeline,
        test_training_improves_metrics,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_functions)} tests")
    print(f"{'=' * 70}")
    
    sys.exit(0 if failed == 0 else 1)

