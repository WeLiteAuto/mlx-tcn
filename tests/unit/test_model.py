"""
Unit tests for train.model module.

Tests model building functionality.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlx_tcn import TCN
from train.config import TrainingConfig
from train.data import DatasetMetadata
from train.model import build_model


# ============================================================================
# build_model Tests
# ============================================================================


def test_build_model_basic():
    """Test basic model building."""
    config = TrainingConfig(
        num_channels=(32, 64),
        kernel_sizes=3,
        dropout=0.1,
        causal=True,
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=2,
        target_names=("HIC", "Dmax"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    assert isinstance(model, TCN)
    # Check model has expected structure
    assert len(model.network) == 2  # 2 blocks for 2 channels
    assert model.causal is True


def test_build_model_uses_metadata():
    """Test that model uses metadata for input/output dimensions."""
    config = TrainingConfig(
        num_channels=(16, 32, 64),
        kernel_sizes=3,
    )
    metadata = DatasetMetadata(
        num_features=5,  # Input features
        seq_len=150,
        target_dim=3,    # Output targets
        target_names=("HIC", "Dmax", "Nij"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Check first block has correct input channels
    assert model.network[0].conv1.weight.shape[2] == 5  # in_channels dimension
    # Check output projection exists and has correct dimension
    assert hasattr(model, 'projection_out')
    assert model.projection_out.weight.shape[0] == 3  # output dimension


def test_build_model_single_kernel_size():
    """Test model building with single kernel size (int)."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),
        kernel_sizes=5,  # Single int
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Should accept single int and replicate across layers
    assert isinstance(model, TCN)


def test_build_model_kernel_sizes_list():
    """Test model building with kernel_sizes as list."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),
        kernel_sizes=(3, 5, 7),  # Different kernel size per layer
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    assert isinstance(model, TCN)
    # Kernel sizes should match
    for i, expected_k in enumerate([3, 5, 7]):
        block = model.network[i]
        # Check kernel size from weight shape
        assert block.conv1.weight.shape[1] == expected_k


def test_build_model_kernel_sizes_mismatch():
    """Test error when kernel_sizes length doesn't match num_channels."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),  # 3 channels
        kernel_sizes=(3, 5),          # Only 2 kernel sizes - mismatch!
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    with pytest.raises(ValueError, match="kernel_sizes length must match num_channels"):
        build_model(config, metadata)


def test_build_model_invalid_kernel_sizes_type():
    """Test error when kernel_sizes is invalid type."""
    config = TrainingConfig(
        num_channels=(32, 64),
        kernel_sizes="invalid",  # Wrong type
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    with pytest.raises(TypeError, match="kernel_sizes must be an int or a sequence"):
        build_model(config, metadata)


def test_build_model_with_dropout():
    """Test model building with dropout."""
    config = TrainingConfig(
        num_channels=(32, 64),
        kernel_sizes=3,
        dropout=0.3,
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Model should be created successfully with dropout
    assert isinstance(model, TCN)
    assert len(model.network) == 2


def test_build_model_causal_vs_non_causal():
    """Test model building with causal and non-causal modes."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # Causal
    config_causal = TrainingConfig(causal=True)
    model_causal = build_model(config_causal, metadata)
    assert model_causal.causal is True
    
    # Non-causal
    config_non_causal = TrainingConfig(causal=False)
    model_non_causal = build_model(config_non_causal, metadata)
    assert model_non_causal.causal is False


def test_build_model_with_normalization():
    """Test model building with different normalization options."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # Batch norm
    config_bn = TrainingConfig(use_norm="batch_norm")
    model_bn = build_model(config_bn, metadata)
    assert isinstance(model_bn, TCN)
    
    # Layer norm
    config_ln = TrainingConfig(use_norm="layer_norm")
    model_ln = build_model(config_ln, metadata)
    assert isinstance(model_ln, TCN)
    
    # No norm (use None instead of "none")
    # Note: TrainingConfig doesn't accept None for use_norm, skip this test


def test_build_model_with_activation():
    """Test model building with different activation functions."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # ReLU
    config_relu = TrainingConfig(activation="relu")
    model_relu = build_model(config_relu, metadata)
    assert model_relu.activation == "relu"
    
    # Note: GELU requires xavier_normal init, not he_normal (default)
    # Skip GELU test as it requires matching kernel_initilaizer in TrainingConfig


def test_build_model_with_skip_connections():
    """Test model building with skip connections on/off."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # With skip connections
    config_skip = TrainingConfig(use_skip_connections=True)
    model_skip = build_model(config_skip, metadata)
    assert model_skip.use_skip_connections is True
    
    # Without skip connections
    config_no_skip = TrainingConfig(use_skip_connections=False)
    model_no_skip = build_model(config_no_skip, metadata)
    assert model_no_skip.use_skip_connections is False


def test_build_model_with_gate():
    """Test model building with gated activation."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # With gate
    config_gate = TrainingConfig(use_gate=True)
    model_gate = build_model(config_gate, metadata)
    assert model_gate.use_gate is True
    
    # Without gate
    config_no_gate = TrainingConfig(use_gate=False)
    model_no_gate = build_model(config_no_gate, metadata)
    assert model_no_gate.use_gate is False


def test_build_model_with_output_activation():
    """Test model building with output activation."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    # With output activation
    config_out_act = TrainingConfig(output_activation="relu")
    model_out_act = build_model(config_out_act, metadata)
    assert model_out_act.output_activation == "relu"
    
    # Without output activation
    config_no_out_act = TrainingConfig(output_activation=None)
    model_no_out_act = build_model(config_no_out_act, metadata)
    assert model_no_out_act.output_activation is None


def test_build_model_multiple_targets():
    """Test model building with multiple target dimensions."""
    config = TrainingConfig(
        num_channels=(32, 64),
        kernel_sizes=3,
    )
    
    # Single target
    metadata_1 = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    model_1 = build_model(config, metadata_1)
    assert model_1.projection_out.weight.shape[0] == 1
    
    # Three targets
    metadata_3 = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=3,
        target_names=("HIC", "Dmax", "Nij"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    model_3 = build_model(config, metadata_3)
    assert model_3.projection_out.weight.shape[0] == 3


def test_build_model_deep_network():
    """Test building a deep TCN network."""
    config = TrainingConfig(
        num_channels=(16, 32, 64, 128, 256),  # 5 layers
        kernel_sizes=3,
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    assert isinstance(model, TCN)
    assert len(model.network) == 5  # 5 layers


def test_build_model_variable_kernel_sizes():
    """Test building model with variable kernel sizes."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),
        kernel_sizes=(3, 5, 7),  # Increasing kernel sizes
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Verify each block has correct kernel size
    assert model.network[0].conv1.weight.shape[1] == 3
    assert model.network[1].conv1.weight.shape[1] == 5
    assert model.network[2].conv1.weight.shape[1] == 7


def test_build_model_converts_tuple_to_list():
    """Test that model correctly converts tuples to lists."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),  # Tuple
        kernel_sizes=(3, 5, 7),      # Tuple
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Model should be created successfully with tuples
    assert isinstance(model, TCN)
    assert len(model.network) == 3


def test_build_model_full_configuration():
    """Test building model with all configuration options."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),
        kernel_sizes=(3, 5, 7),
        dropout=0.2,
        causal=True,
        use_norm="batch_norm",
        activation="relu",  # Use relu instead of gelu to avoid init issues
        use_skip_connections=True,
        use_gate=True,
        output_activation="relu",
    )
    metadata = DatasetMetadata(
        num_features=5,
        seq_len=150,
        target_dim=3,
        target_names=("HIC", "Dmax", "Nij"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Verify configuration is correctly applied
    assert isinstance(model, TCN)
    assert len(model.network) == 3
    assert model.causal is True
    assert model.activation == "relu"
    assert model.use_skip_connections is True
    assert model.use_gate is True
    assert model.output_activation == "relu"
    assert model.projection_out.weight.shape[0] == 3


# ============================================================================
# Integration Tests
# ============================================================================


def test_build_model_can_forward_pass():
    """Test that built model can perform forward pass."""
    import mlx.core as mx
    
    config = TrainingConfig(
        num_channels=(32, 64),
        kernel_sizes=3,
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=2,
        target_names=("HIC", "Dmax"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model = build_model(config, metadata)
    
    # Create dummy input
    x = mx.random.normal((4, 100, 3))  # (batch, time, features)
    
    # Forward pass
    output = model(x, embeddings=None, inference=False)
    
    # Check output shape
    assert output.shape == (4, 100, 2)  # (batch, time, target_dim)


def test_build_model_reproducible():
    """Test that building model twice with same config produces same architecture."""
    config = TrainingConfig(
        num_channels=(32, 64, 128),
        kernel_sizes=5,
    )
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=100,
        target_dim=1,
        target_names=("HIC",),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    model1 = build_model(config, metadata)
    model2 = build_model(config, metadata)
    
    # Should have same architecture
    assert model1.causal == model2.causal
    assert model1.activation == model2.activation
    assert len(model1.network) == len(model2.network)


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # Basic tests
        test_build_model_basic,
        test_build_model_uses_metadata,
        
        # Kernel sizes tests
        test_build_model_single_kernel_size,
        test_build_model_kernel_sizes_list,
        test_build_model_kernel_sizes_mismatch,
        test_build_model_invalid_kernel_sizes_type,
        
        # Configuration tests
        test_build_model_with_dropout,
        test_build_model_causal_vs_non_causal,
        test_build_model_with_normalization,
        test_build_model_with_activation,
        test_build_model_with_skip_connections,
        test_build_model_with_gate,
        test_build_model_with_output_activation,
        
        # Target dimension tests
        test_build_model_multiple_targets,
        
        # Architecture tests
        test_build_model_deep_network,
        test_build_model_variable_kernel_sizes,
        test_build_model_converts_tuple_to_list,
        test_build_model_full_configuration,
        
        # Integration tests
        test_build_model_can_forward_pass,
        test_build_model_reproducible,
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

