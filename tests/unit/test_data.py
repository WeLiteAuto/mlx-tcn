"""
Unit tests for train.data module.

Tests dataset loading, preprocessing, and batch iteration functionality.
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.config import TrainingConfig
from train.data import (
    DatasetBundle,
    DatasetMetadata,
    WaveformDataset,
    iter_batches,
    load_datasets,
)


# ============================================================================
# DatasetMetadata Tests
# ============================================================================


def test_dataset_metadata_init():
    """Test DatasetMetadata initialization."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=150,
        target_dim=2,
        target_names=("HIC", "Dmax"),
        num_continuous_params=14,
        num_discrete_params=4,
        discrete_indices=(3, 9, 11, 14),
    )
    
    assert metadata.num_features == 3
    assert metadata.seq_len == 150
    assert metadata.target_dim == 2
    assert metadata.target_names == ("HIC", "Dmax")
    assert metadata.num_continuous_params == 14
    assert metadata.num_discrete_params == 4
    assert metadata.discrete_indices == (3, 9, 11, 14)


def test_dataset_metadata_tuple_types():
    """Test that metadata stores tuples not lists."""
    metadata = DatasetMetadata(
        num_features=3,
        seq_len=150,
        target_dim=1,
        target_names=("label1",),
        num_continuous_params=10,
        num_discrete_params=2,
        discrete_indices=(0, 1),
    )
    
    assert isinstance(metadata.target_names, tuple)
    assert isinstance(metadata.discrete_indices, tuple)


# ============================================================================
# WaveformDataset Tests
# ============================================================================


def test_waveform_dataset_init():
    """Test WaveformDataset initialization."""
    features = np.random.randn(10, 150, 3).astype(np.float32)
    labels = np.random.randn(10, 2).astype(np.float32)
    cont_params = np.random.randn(10, 14).astype(np.float32)
    disc_params = np.random.randint(0, 5, (10, 4)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    assert dataset.features.shape == (10, 150, 3)
    assert dataset.labels.shape == (10, 2)
    assert dataset.continuous_params.shape == (10, 14)
    assert dataset.discrete_params.shape == (10, 4)


def test_waveform_dataset_dtype_conversion():
    """Test automatic dtype conversion in __post_init__."""
    # Create with wrong dtypes
    features = np.random.randn(5, 10, 2).astype(np.float64)
    labels = np.random.randn(5, 1).astype(np.float64)
    cont_params = np.random.randn(5, 3).astype(np.float64)
    disc_params = np.random.randint(0, 3, (5, 2)).astype(np.int64)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    # Should be converted to correct dtypes
    assert dataset.features.dtype == np.float32
    assert dataset.labels.dtype == np.float32
    assert dataset.continuous_params.dtype == np.float32
    assert dataset.discrete_params.dtype == np.int32


def test_waveform_dataset_len():
    """Test __len__ returns correct number of samples."""
    features = np.random.randn(42, 100, 5).astype(np.float32)
    labels = np.random.randn(42, 1).astype(np.float32)
    cont_params = np.random.randn(42, 10).astype(np.float32)
    disc_params = np.random.randint(0, 2, (42, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    assert len(dataset) == 42


def test_waveform_dataset_get_batch():
    """Test get_batch retrieves correct samples."""
    features = np.arange(10 * 5 * 2).reshape(10, 5, 2).astype(np.float32)
    labels = np.arange(10 * 1).reshape(10, 1).astype(np.float32)
    cont_params = np.arange(10 * 3).reshape(10, 3).astype(np.float32)
    disc_params = np.arange(10 * 2).reshape(10, 2).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    indices = np.array([0, 2, 5])
    batch_f, batch_l, batch_c, batch_d = dataset.get_batch(indices)
    
    assert batch_f.shape == (3, 5, 2)
    assert batch_l.shape == (3, 1)
    assert batch_c.shape == (3, 3)
    assert batch_d.shape == (3, 2)
    
    # Check values
    np.testing.assert_array_equal(batch_f[0], features[0])
    np.testing.assert_array_equal(batch_f[1], features[2])
    np.testing.assert_array_equal(batch_f[2], features[5])


def test_waveform_dataset_empty():
    """Test WaveformDataset with empty arrays."""
    features = np.zeros((0, 10, 2), dtype=np.float32)
    labels = np.zeros((0, 1), dtype=np.float32)
    cont_params = np.zeros((0, 3), dtype=np.float32)
    disc_params = np.zeros((0, 2), dtype=np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    assert len(dataset) == 0


# ============================================================================
# DatasetBundle Tests
# ============================================================================


def test_dataset_bundle_init():
    """Test DatasetBundle initialization."""
    features = np.random.randn(10, 5, 2).astype(np.float32)
    labels = np.random.randn(10, 1).astype(np.float32)
    cont_params = np.random.randn(10, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (10, 2)).astype(np.int32)
    
    train_ds = WaveformDataset(features[:8], labels[:8], cont_params[:8], disc_params[:8])
    val_ds = WaveformDataset(features[8:], labels[8:], cont_params[8:], disc_params[8:])
    metadata = DatasetMetadata(2, 5, 1, ("label",), 3, 2, (0, 1))
    
    bundle = DatasetBundle(train=train_ds, val=val_ds, metadata=metadata)
    
    assert bundle.train is train_ds
    assert bundle.val is val_ds
    assert bundle.metadata is metadata


# ============================================================================
# load_datasets Tests
# ============================================================================


def create_test_npz_files(tmp_dir, n_samples=20, n_channels=3, n_time=50, n_params=18):
    """Helper to create test npz files."""
    input_path = tmp_dir / "test_input.npz"
    label_path = tmp_dir / "test_labels.npz"
    
    # Create input data
    waveforms = np.random.randn(n_samples, n_channels, n_time).astype(np.float64)
    params = np.random.randn(n_samples, n_params).astype(np.float32)
    np.savez(input_path, waveforms=waveforms, params=params)
    
    # Create label data
    hic = np.random.randn(n_samples).astype(np.float64)
    dmax = np.random.randn(n_samples).astype(np.float64)
    nij = np.random.randn(n_samples).astype(np.float64)
    np.savez(label_path, HIC=hic, Dmax=dmax, Nij=nij)
    
    return input_path, label_path


def test_load_datasets_basic():
    """Test basic dataset loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path, n_samples=20)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC", "Dmax", "Nij"),
            train_split=0.8,
            seed=42,
        )
        
        bundle = load_datasets(config)
        
        # Check bundle structure
        assert isinstance(bundle, DatasetBundle)
        assert isinstance(bundle.train, WaveformDataset)
        assert isinstance(bundle.val, WaveformDataset)
        assert isinstance(bundle.metadata, DatasetMetadata)
        
        # Check split sizes
        assert len(bundle.train) == 16  # 80% of 20
        assert len(bundle.val) == 4     # 20% of 20
        
        # Check metadata
        assert bundle.metadata.num_features == 3
        assert bundle.metadata.seq_len == 50
        assert bundle.metadata.target_dim == 3
        assert bundle.metadata.target_names == ("HIC", "Dmax", "Nij")


def test_load_datasets_shape_transformation():
    """Test that waveforms are transposed from (N,C,T) to (N,T,C)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        n_samples, n_channels, n_time = 10, 3, 50
        input_path, label_path = create_test_npz_files(
            tmp_path, n_samples=n_samples, n_channels=n_channels, n_time=n_time
        )
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
            train_split=0.8,
        )
        
        bundle = load_datasets(config)
        
        # Original: (N, C, T), Transformed: (N, T, C)
        assert bundle.train.features.shape[1] == n_time
        assert bundle.train.features.shape[2] == n_channels


def test_load_datasets_params_split():
    """Test that params are correctly split into continuous and discrete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path, n_params=18)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
            train_split=0.8,
        )
        
        bundle = load_datasets(config)
        
        # Discrete indices: (3, 9, 11, 14) -> 4 discrete params
        # Continuous: 18 - 4 = 14 params
        assert bundle.metadata.num_discrete_params == 4
        assert bundle.metadata.num_continuous_params == 14
        assert bundle.metadata.discrete_indices == (3, 9, 11, 14)
        
        # Check actual shapes
        assert bundle.train.discrete_params.shape[1] == 4
        assert bundle.train.continuous_params.shape[1] == 14


def test_load_datasets_dtypes():
    """Test that loaded data has correct dtypes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC", "Dmax"),
            train_split=0.8,
        )
        
        bundle = load_datasets(config)
        
        assert bundle.train.features.dtype == np.float32
        assert bundle.train.labels.dtype == np.float32
        assert bundle.train.continuous_params.dtype == np.float32
        assert bundle.train.discrete_params.dtype == np.int32


def test_load_datasets_reproducible_with_seed():
    """Test that loading with same seed produces same split."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
            train_split=0.8,
            seed=123,
        )
        
        bundle1 = load_datasets(config)
        bundle2 = load_datasets(config)
        
        # Same samples in train/val splits
        np.testing.assert_array_equal(bundle1.train.features, bundle2.train.features)
        np.testing.assert_array_equal(bundle1.val.features, bundle2.val.features)


def test_load_datasets_different_seeds():
    """Test that different seeds produce different splits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path, n_samples=50)
        
        config1 = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
            train_split=0.8,
            seed=42,
        )
        config2 = config1.with_updates(seed=99)
        
        bundle1 = load_datasets(config1)
        bundle2 = load_datasets(config2)
        
        # Different samples in splits
        assert not np.array_equal(bundle1.train.features, bundle2.train.features)


def test_load_datasets_missing_waveforms_key():
    """Test error when 'waveforms' key is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "bad_input.npz"
        label_path = tmp_path / "labels.npz"
        
        # Missing 'waveforms' key
        params = np.random.randn(10, 18)
        np.savez(input_path, params=params)
        np.savez(label_path, HIC=np.random.randn(10))
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
        )
        
        with pytest.raises(KeyError, match="Expected key 'waveforms'"):
            load_datasets(config)


def test_load_datasets_missing_params_key():
    """Test error when 'params' key is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "bad_input.npz"
        label_path = tmp_path / "labels.npz"
        
        # Missing 'params' key
        waveforms = np.random.randn(10, 3, 50)
        np.savez(input_path, waveforms=waveforms)
        np.savez(label_path, HIC=np.random.randn(10))
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
        )
        
        with pytest.raises(KeyError, match="Expected key 'params'"):
            load_datasets(config)


def test_load_datasets_missing_label_key():
    """Test error when requested label key is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC", "NonExistent"),  # NonExistent doesn't exist
        )
        
        with pytest.raises(KeyError, match="Missing target keys"):
            load_datasets(config)


def test_load_datasets_invalid_waveform_shape():
    """Test error when waveforms don't have 3 dimensions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "bad_input.npz"
        label_path = tmp_path / "labels.npz"
        
        # Wrong shape: 2D instead of 3D
        waveforms = np.random.randn(10, 50)
        params = np.random.randn(10, 18)
        np.savez(input_path, waveforms=waveforms, params=params)
        np.savez(label_path, HIC=np.random.randn(10))
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
        )
        
        with pytest.raises(ValueError, match="Expected waveforms to have shape"):
            load_datasets(config)


def test_load_datasets_invalid_params_shape():
    """Test error when params don't have 2 dimensions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "bad_input.npz"
        label_path = tmp_path / "labels.npz"
        
        waveforms = np.random.randn(10, 3, 50)
        # Wrong shape: 1D instead of 2D
        params = np.random.randn(10)
        np.savez(input_path, waveforms=waveforms, params=params)
        np.savez(label_path, HIC=np.random.randn(10))
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
        )
        
        with pytest.raises(ValueError, match="Expected params to have shape"):
            load_datasets(config)


def test_load_datasets_invalid_train_split():
    """Test error when train_split is out of range."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path)
        
        # train_split = 0 (invalid)
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
            train_split=0.0,
        )
        
        with pytest.raises(ValueError, match="train_split must be between 0 and 1"):
            load_datasets(config)
        
        # train_split = 1 (invalid)
        config2 = config.with_updates(train_split=1.0)
        with pytest.raises(ValueError, match="train_split must be between 0 and 1"):
            load_datasets(config2)


def test_load_datasets_discrete_index_out_of_range():
    """Test error when discrete indices exceed params dimensions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.npz"
        label_path = tmp_path / "labels.npz"
        
        # Only 5 params, but discrete_indices expects at least 15
        waveforms = np.random.randn(10, 3, 50)
        params = np.random.randn(10, 5)  # Too few params
        np.savez(input_path, waveforms=waveforms, params=params)
        np.savez(label_path, HIC=np.random.randn(10))
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC",),
        )
        
        with pytest.raises(ValueError, match="Discrete feature index out of range"):
            load_datasets(config)


# ============================================================================
# iter_batches Tests
# ============================================================================


def test_iter_batches_basic():
    """Test basic batch iteration."""
    features = np.random.randn(10, 5, 2).astype(np.float32)
    labels = np.random.randn(10, 1).astype(np.float32)
    cont_params = np.random.randn(10, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (10, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=3, shuffle=False))
    
    # 10 samples / batch_size 3 = 4 batches (3, 3, 3, 1)
    assert len(batches) == 4
    
    # Check batch shapes
    batch_x, batch_y, batch_c, batch_d = batches[0]
    assert batch_x.shape == (3, 5, 2)
    assert batch_y.shape == (3, 1)
    assert batch_c.shape == (3, 3)
    assert batch_d.shape == (3, 2)
    
    # Last batch has 1 sample
    assert batches[3][0].shape[0] == 1


def test_iter_batches_returns_mlx_arrays():
    """Test that iter_batches returns MLX arrays."""
    features = np.random.randn(5, 3, 2).astype(np.float32)
    labels = np.random.randn(5, 1).astype(np.float32)
    cont_params = np.random.randn(5, 2).astype(np.float32)
    disc_params = np.random.randint(0, 2, (5, 1)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=2, shuffle=False))
    batch_x, batch_y, batch_c, batch_d = batches[0]
    
    assert isinstance(batch_x, mx.array)
    assert isinstance(batch_y, mx.array)
    assert isinstance(batch_c, mx.array)
    assert isinstance(batch_d, mx.array)


def test_iter_batches_shuffle_false():
    """Test that shuffle=False preserves order."""
    features = np.arange(10 * 2 * 1).reshape(10, 2, 1).astype(np.float32)
    labels = np.arange(10).reshape(10, 1).astype(np.float32)
    cont_params = np.random.randn(10, 1).astype(np.float32)
    disc_params = np.random.randint(0, 2, (10, 1)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=10, shuffle=False))
    batch_x, batch_y, _, _ = batches[0]
    
    # Should be in original order
    np.testing.assert_array_equal(batch_y, np.arange(10).reshape(10, 1))


def test_iter_batches_shuffle_true():
    """Test that shuffle=True with seed is reproducible."""
    features = np.random.randn(20, 5, 2).astype(np.float32)
    labels = np.random.randn(20, 1).astype(np.float32)
    cont_params = np.random.randn(20, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (20, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    # Same seed should produce same order
    batches1 = list(iter_batches(dataset, batch_size=5, shuffle=True, seed=42))
    batches2 = list(iter_batches(dataset, batch_size=5, shuffle=True, seed=42))
    
    for (b1_x, _, _, _), (b2_x, _, _, _) in zip(batches1, batches2):
        np.testing.assert_array_equal(b1_x, b2_x)


def test_iter_batches_different_seeds():
    """Test that different seeds produce different orders."""
    features = np.random.randn(50, 5, 2).astype(np.float32)
    labels = np.random.randn(50, 1).astype(np.float32)
    cont_params = np.random.randn(50, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (50, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches1 = list(iter_batches(dataset, batch_size=10, shuffle=True, seed=1))
    batches2 = list(iter_batches(dataset, batch_size=10, shuffle=True, seed=2))
    
    # At least first batch should be different
    b1_x, _, _, _ = batches1[0]
    b2_x, _, _, _ = batches2[0]
    
    assert not np.array_equal(b1_x, b2_x)


def test_iter_batches_empty_dataset():
    """Test iter_batches with empty dataset."""
    features = np.zeros((0, 5, 2), dtype=np.float32)
    labels = np.zeros((0, 1), dtype=np.float32)
    cont_params = np.zeros((0, 3), dtype=np.float32)
    disc_params = np.zeros((0, 2), dtype=np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=10))
    
    assert len(batches) == 0


def test_iter_batches_single_sample():
    """Test iter_batches with single sample."""
    features = np.random.randn(1, 5, 2).astype(np.float32)
    labels = np.random.randn(1, 1).astype(np.float32)
    cont_params = np.random.randn(1, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (1, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=10))
    
    assert len(batches) == 1
    batch_x, _, _, _ = batches[0]
    assert batch_x.shape[0] == 1


def test_iter_batches_exact_division():
    """Test iter_batches when dataset size divides evenly by batch size."""
    features = np.random.randn(20, 5, 2).astype(np.float32)
    labels = np.random.randn(20, 1).astype(np.float32)
    cont_params = np.random.randn(20, 3).astype(np.float32)
    disc_params = np.random.randint(0, 2, (20, 2)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=5, shuffle=False))
    
    # 20 / 5 = 4 batches, all same size
    assert len(batches) == 4
    for batch_x, _, _, _ in batches:
        assert batch_x.shape[0] == 5


def test_iter_batches_large_batch_size():
    """Test iter_batches when batch_size > dataset size."""
    features = np.random.randn(5, 3, 2).astype(np.float32)
    labels = np.random.randn(5, 1).astype(np.float32)
    cont_params = np.random.randn(5, 2).astype(np.float32)
    disc_params = np.random.randint(0, 2, (5, 1)).astype(np.int32)
    
    dataset = WaveformDataset(features, labels, cont_params, disc_params)
    
    batches = list(iter_batches(dataset, batch_size=100, shuffle=False))
    
    # Should have 1 batch with all 5 samples
    assert len(batches) == 1
    batch_x, _, _, _ = batches[0]
    assert batch_x.shape[0] == 5


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline():
    """Test complete data loading and batch iteration pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path, label_path = create_test_npz_files(tmp_path, n_samples=50)
        
        config = TrainingConfig(
            data_dir=tmp_path,
            input_file=input_path.name,
            label_file=label_path.name,
            label_keys=("HIC", "Dmax"),
            train_split=0.8,
            batch_size=8,
            seed=42,
        )
        
        # Load datasets
        bundle = load_datasets(config)
        
        # Iterate training batches
        train_batches = list(iter_batches(
            bundle.train,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed
        ))
        
        # Check we got expected number of batches
        expected_batches = (len(bundle.train) + config.batch_size - 1) // config.batch_size
        assert len(train_batches) == expected_batches
        
        # Check batch contents
        for batch_x, batch_y, batch_c, batch_d in train_batches:
            assert isinstance(batch_x, mx.array)
            assert batch_x.shape[1] == bundle.metadata.seq_len
            assert batch_x.shape[2] == bundle.metadata.num_features
            assert batch_y.shape[1] == bundle.metadata.target_dim


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests when executed directly."""
    test_functions = [
        # DatasetMetadata tests
        test_dataset_metadata_init,
        test_dataset_metadata_tuple_types,
        
        # WaveformDataset tests
        test_waveform_dataset_init,
        test_waveform_dataset_dtype_conversion,
        test_waveform_dataset_len,
        test_waveform_dataset_get_batch,
        test_waveform_dataset_empty,
        
        # DatasetBundle tests
        test_dataset_bundle_init,
        
        # load_datasets tests
        test_load_datasets_basic,
        test_load_datasets_shape_transformation,
        test_load_datasets_params_split,
        test_load_datasets_dtypes,
        test_load_datasets_reproducible_with_seed,
        test_load_datasets_different_seeds,
        test_load_datasets_missing_waveforms_key,
        test_load_datasets_missing_params_key,
        test_load_datasets_missing_label_key,
        test_load_datasets_invalid_waveform_shape,
        test_load_datasets_invalid_params_shape,
        test_load_datasets_invalid_train_split,
        test_load_datasets_discrete_index_out_of_range,
        
        # iter_batches tests
        test_iter_batches_basic,
        test_iter_batches_returns_mlx_arrays,
        test_iter_batches_shuffle_false,
        test_iter_batches_shuffle_true,
        test_iter_batches_different_seeds,
        test_iter_batches_empty_dataset,
        test_iter_batches_single_sample,
        test_iter_batches_exact_division,
        test_iter_batches_large_batch_size,
        
        # Integration tests
        test_full_pipeline,
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

