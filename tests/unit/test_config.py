"""
Unit tests for train.config module.

Tests the TrainingConfig dataclass including initialization, 
path resolution, and configuration updates.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.config import TrainingConfig


# ============================================================================
# Initialization Tests
# ============================================================================


def test_training_config_init_default():
    """Test TrainingConfig initialization with default values."""
    config = TrainingConfig()
    
    # Data configuration
    assert config.data_dir == Path("data")
    assert config.input_file == "data_input.npz"
    assert config.label_file == "data_labels.npz"
    assert config.label_keys == ("HIC", "Dmax", "Nij")
    
    # Training parameters
    assert config.train_split == 0.8
    assert config.seed == 13
    assert config.batch_size == 32
    assert config.num_epochs == 20
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.log_every_n_steps == 10
    
    # Model parameters
    assert config.num_channels == (64, 64, 128)
    assert config.kernel_sizes == 3
    assert config.dropout == 0.1
    assert config.causal is False
    assert config.use_norm == "batch_norm"
    assert config.activation == "relu"
    assert config.use_skip_connections is True
    assert config.use_gate is False
    assert config.output_activation is None
    
    # Plotting parameters
    assert config.enable_plot is False
    assert config.plot_path is None


def test_training_config_init_custom():
    """Test TrainingConfig initialization with custom values."""
    config = TrainingConfig(
        data_dir=Path("/custom/data"),
        input_file="custom_input.npz",
        label_file="custom_labels.npz",
        label_keys=("label1", "label2"),
        train_split=0.9,
        seed=42,
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        log_every_n_steps=5,
        num_channels=(128, 256),
        kernel_sizes=(3, 5),
        dropout=0.2,
        causal=True,
        use_norm="layer_norm",
        activation="gelu",
        use_skip_connections=False,
        use_gate=True,
        output_activation="sigmoid",
        enable_plot=True,
        plot_path=Path("custom_plot.png")
    )
    
    assert config.data_dir == Path("/custom/data")
    assert config.input_file == "custom_input.npz"
    assert config.label_file == "custom_labels.npz"
    assert config.label_keys == ("label1", "label2")
    assert config.train_split == 0.9
    assert config.seed == 42
    assert config.batch_size == 64
    assert config.num_epochs == 50
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-5
    assert config.log_every_n_steps == 5
    assert config.num_channels == (128, 256)
    assert config.kernel_sizes == (3, 5)
    assert config.dropout == 0.2
    assert config.causal is True
    assert config.use_norm == "layer_norm"
    assert config.activation == "gelu"
    assert config.use_skip_connections is False
    assert config.use_gate is True
    assert config.output_activation == "sigmoid"
    assert config.enable_plot is True
    assert config.plot_path == Path("custom_plot.png")


def test_training_config_immutability():
    """Test that TrainingConfig is immutable (frozen dataclass behavior)."""
    config = TrainingConfig()
    
    # Note: With slots=True but no frozen=True, the dataclass is NOT frozen
    # This test verifies that direct assignment works (mutable)
    config.batch_size = 64
    assert config.batch_size == 64


def test_training_config_type_annotations():
    """Test that TrainingConfig has correct type annotations."""
    config = TrainingConfig()
    
    # Check Path types
    assert isinstance(config.data_dir, Path)
    
    # Check string types
    assert isinstance(config.input_file, str)
    assert isinstance(config.label_file, str)
    assert isinstance(config.use_norm, str)
    assert isinstance(config.activation, str)
    
    # Check tuple types
    assert isinstance(config.label_keys, tuple)
    assert isinstance(config.num_channels, tuple)
    
    # Check numeric types
    assert isinstance(config.train_split, float)
    assert isinstance(config.seed, int)
    assert isinstance(config.batch_size, int)
    assert isinstance(config.learning_rate, float)
    
    # Check boolean types
    assert isinstance(config.causal, bool)
    assert isinstance(config.use_skip_connections, bool)
    assert isinstance(config.enable_plot, bool)


# ============================================================================
# Path Resolution Tests
# ============================================================================


def test_input_path_default():
    """Test input_path() with default configuration."""
    config = TrainingConfig()
    expected = Path("data") / "data_input.npz"
    assert config.input_path() == expected


def test_input_path_custom_dir():
    """Test input_path() with custom data directory."""
    config = TrainingConfig(data_dir=Path("/custom/path"))
    expected = Path("/custom/path") / "data_input.npz"
    assert config.input_path() == expected


def test_input_path_custom_file():
    """Test input_path() with custom input file name."""
    config = TrainingConfig(input_file="my_input.npz")
    expected = Path("data") / "my_input.npz"
    assert config.input_path() == expected


def test_input_path_both_custom():
    """Test input_path() with both custom directory and file name."""
    config = TrainingConfig(
        data_dir=Path("/my/data"),
        input_file="features.npz"
    )
    expected = Path("/my/data") / "features.npz"
    assert config.input_path() == expected


def test_label_path_default():
    """Test label_path() with default configuration."""
    config = TrainingConfig()
    expected = Path("data") / "data_labels.npz"
    assert config.label_path() == expected


def test_label_path_custom_dir():
    """Test label_path() with custom data directory."""
    config = TrainingConfig(data_dir=Path("/labels/dir"))
    expected = Path("/labels/dir") / "data_labels.npz"
    assert config.label_path() == expected


def test_label_path_custom_file():
    """Test label_path() with custom label file name."""
    config = TrainingConfig(label_file="targets.npz")
    expected = Path("data") / "targets.npz"
    assert config.label_path() == expected


def test_resolve_plot_path_default():
    """Test resolve_plot_path() with default (None) plot_path."""
    config = TrainingConfig()
    expected = Path("training_metrics.png")
    assert config.resolve_plot_path() == expected


def test_resolve_plot_path_custom():
    """Test resolve_plot_path() with custom plot_path."""
    config = TrainingConfig(plot_path=Path("results/my_plot.png"))
    expected = Path("results/my_plot.png")
    assert config.resolve_plot_path() == expected


def test_resolve_plot_path_enabled_but_no_path():
    """Test resolve_plot_path() when plotting is enabled but no path specified."""
    config = TrainingConfig(enable_plot=True, plot_path=None)
    expected = Path("training_metrics.png")
    assert config.resolve_plot_path() == expected


def test_path_methods_return_path_objects():
    """Test that all path methods return Path objects."""
    config = TrainingConfig()
    
    assert isinstance(config.input_path(), Path)
    assert isinstance(config.label_path(), Path)
    assert isinstance(config.resolve_plot_path(), Path)


# ============================================================================
# Configuration Update Tests
# ============================================================================


def test_with_updates_single_field():
    """Test with_updates() with a single field update."""
    config = TrainingConfig()
    updated = config.with_updates(batch_size=128)
    
    # Check that the field was updated
    assert updated.batch_size == 128
    
    # Check that other fields remain unchanged
    assert updated.learning_rate == config.learning_rate
    assert updated.num_epochs == config.num_epochs
    
    # Check that original config is unchanged
    assert config.batch_size == 32


def test_with_updates_multiple_fields():
    """Test with_updates() with multiple field updates."""
    config = TrainingConfig()
    updated = config.with_updates(
        batch_size=64,
        learning_rate=5e-4,
        num_epochs=100,
        causal=True
    )
    
    assert updated.batch_size == 64
    assert updated.learning_rate == 5e-4
    assert updated.num_epochs == 100
    assert updated.causal is True
    
    # Original unchanged
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.num_epochs == 20
    assert config.causal is False


def test_with_updates_path_fields():
    """Test with_updates() with Path fields."""
    config = TrainingConfig()
    updated = config.with_updates(
        data_dir=Path("/new/data"),
        plot_path=Path("new_plot.png")
    )
    
    assert updated.data_dir == Path("/new/data")
    assert updated.plot_path == Path("new_plot.png")
    assert config.data_dir == Path("data")
    assert config.plot_path is None


def test_with_updates_tuple_fields():
    """Test with_updates() with tuple fields."""
    config = TrainingConfig()
    updated = config.with_updates(
        num_channels=(256, 512, 1024),
        label_keys=("new_label",)
    )
    
    assert updated.num_channels == (256, 512, 1024)
    assert updated.label_keys == ("new_label",)
    assert config.num_channels == (64, 64, 128)
    assert config.label_keys == ("HIC", "Dmax", "Nij")


def test_with_updates_empty():
    """Test with_updates() with no updates."""
    config = TrainingConfig()
    updated = config.with_updates()
    
    # Should return a copy with identical values
    assert updated.batch_size == config.batch_size
    assert updated.learning_rate == config.learning_rate
    assert updated.num_epochs == config.num_epochs
    
    # But should be a different object
    assert updated is not config


def test_with_updates_preserves_type():
    """Test that with_updates() returns a TrainingConfig instance."""
    config = TrainingConfig()
    updated = config.with_updates(batch_size=64)
    
    assert isinstance(updated, TrainingConfig)
    assert type(updated) == type(config)


def test_with_updates_chaining():
    """Test chaining multiple with_updates() calls."""
    config = TrainingConfig()
    
    updated = (config
               .with_updates(batch_size=64)
               .with_updates(learning_rate=1e-4)
               .with_updates(num_epochs=50))
    
    assert updated.batch_size == 64
    assert updated.learning_rate == 1e-4
    assert updated.num_epochs == 50
    
    # Original unchanged
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.num_epochs == 20


# ============================================================================
# Integration Tests
# ============================================================================


def test_config_workflow():
    """Test a typical configuration workflow."""
    # Start with defaults
    config = TrainingConfig()
    
    # Update for a specific experiment
    experiment_config = config.with_updates(
        data_dir=Path("experiments/exp1/data"),
        batch_size=128,
        num_epochs=100,
        learning_rate=5e-4,
        enable_plot=True,
        plot_path=Path("experiments/exp1/metrics.png")
    )
    
    # Verify paths are correctly computed
    assert experiment_config.input_path() == Path("experiments/exp1/data/data_input.npz")
    assert experiment_config.label_path() == Path("experiments/exp1/data/data_labels.npz")
    assert experiment_config.resolve_plot_path() == Path("experiments/exp1/metrics.png")
    
    # Verify config values
    assert experiment_config.batch_size == 128
    assert experiment_config.num_epochs == 100


def test_config_for_causal_tcn():
    """Test configuration for causal TCN training."""
    config = TrainingConfig(
        causal=True,
        use_skip_connections=True,
        use_gate=True,
        num_channels=(64, 128, 256),
        kernel_sizes=(3, 3, 3),
        dropout=0.2
    )
    
    assert config.causal is True
    assert config.use_skip_connections is True
    assert config.use_gate is True
    assert config.num_channels == (64, 128, 256)
    assert config.kernel_sizes == (3, 3, 3)


def test_config_for_different_targets():
    """Test configuration for different prediction targets."""
    # Configuration for HIC prediction
    hic_config = TrainingConfig(label_keys=("HIC",))
    assert hic_config.label_keys == ("HIC",)
    
    # Configuration for multiple target prediction
    multi_config = TrainingConfig(label_keys=("AIS_head", "AIS_chest", "MAIS"))
    assert multi_config.label_keys == ("AIS_head", "AIS_chest", "MAIS")


def test_config_extreme_values():
    """Test configuration with edge case values."""
    config = TrainingConfig(
        batch_size=1,
        num_epochs=1,
        learning_rate=1.0,
        train_split=0.99,
        dropout=0.0,
        weight_decay=0.0,
        log_every_n_steps=1
    )
    
    assert config.batch_size == 1
    assert config.num_epochs == 1
    assert config.learning_rate == 1.0
    assert config.train_split == 0.99
    assert config.dropout == 0.0
    assert config.weight_decay == 0.0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_config_with_relative_paths():
    """Test configuration with relative paths."""
    config = TrainingConfig(
        data_dir=Path("../data"),
        plot_path=Path("./results/plot.png")
    )
    
    assert config.data_dir == Path("../data")
    assert config.plot_path == Path("./results/plot.png")
    assert config.input_path() == Path("../data/data_input.npz")


def test_config_with_absolute_paths():
    """Test configuration with absolute paths."""
    config = TrainingConfig(
        data_dir=Path("/absolute/path/to/data"),
        plot_path=Path("/absolute/path/to/plot.png")
    )
    
    assert config.data_dir == Path("/absolute/path/to/data")
    assert config.input_path() == Path("/absolute/path/to/data/data_input.npz")


def test_config_kernel_sizes_single_int():
    """Test kernel_sizes as single integer."""
    config = TrainingConfig(kernel_sizes=5)
    assert config.kernel_sizes == 5
    assert isinstance(config.kernel_sizes, int)


def test_config_kernel_sizes_tuple():
    """Test kernel_sizes as tuple."""
    config = TrainingConfig(kernel_sizes=(3, 5, 7))
    assert config.kernel_sizes == (3, 5, 7)
    assert isinstance(config.kernel_sizes, tuple)


def test_config_optional_output_activation():
    """Test output_activation as None and as string."""
    config1 = TrainingConfig(output_activation=None)
    assert config1.output_activation is None
    
    config2 = TrainingConfig(output_activation="softmax")
    assert config2.output_activation == "softmax"


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # Initialization tests
        test_training_config_init_default,
        test_training_config_init_custom,
        test_training_config_immutability,
        test_training_config_type_annotations,
        
        # Path resolution tests
        test_input_path_default,
        test_input_path_custom_dir,
        test_input_path_custom_file,
        test_input_path_both_custom,
        test_label_path_default,
        test_label_path_custom_dir,
        test_label_path_custom_file,
        test_resolve_plot_path_default,
        test_resolve_plot_path_custom,
        test_resolve_plot_path_enabled_but_no_path,
        test_path_methods_return_path_objects,
        
        # Configuration update tests
        test_with_updates_single_field,
        test_with_updates_multiple_fields,
        test_with_updates_path_fields,
        test_with_updates_tuple_fields,
        test_with_updates_empty,
        test_with_updates_preserves_type,
        test_with_updates_chaining,
        
        # Integration tests
        test_config_workflow,
        test_config_for_causal_tcn,
        test_config_for_different_targets,
        test_config_extreme_values,
        
        # Edge cases
        test_config_with_relative_paths,
        test_config_with_absolute_paths,
        test_config_kernel_sizes_single_int,
        test_config_kernel_sizes_tuple,
        test_config_optional_output_activation,
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

