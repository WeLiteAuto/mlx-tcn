#!/usr/bin/env python3
"""Quick example demonstrating MLX-TCN functionality."""

import mlx.core as mx

from mlx_tcn import BufferIO, TCN

def example_basic_tcn():
    """Example 1: Basic TCN usage."""
    print("="*70)
    print("Example 1: Basic TCN")
    print("="*70)
    
    model = TCN(
        num_inputs=32,
        num_channels=[64, 64, 128],
        kernel_sizes=3,
        dropout=0.1,
        causal=False,
        use_norm="batch_norm",
        activation="relu",
    )
    
    # Forward pass
    batch, seq_len = 4, 100
    x = mx.random.normal((batch, seq_len, 32))
    output = model(x, embeddings=None, inference=False)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of layers: {len(model.network)}")
    print(f"Dilations: {model.dilations}")
    print("✓ Basic TCN works!\n")


def example_tcn_with_skip_connections():
    """Example 2: TCN with skip connections and output projection."""
    print("="*70)
    print("Example 2: TCN with Skip Connections & Output Projection")
    print("="*70)
    
    model = TCN(
        num_inputs=16,
        num_channels=[32, 64, 64],
        kernel_sizes=5,
        use_skip_connections=True,
        output_projection=10,
        output_activation="tanh",
        causal=False,
    )
    
    batch, seq_len = 2, 50
    x = mx.random.normal((batch, seq_len, 16))
    output = model(x, embeddings=None, inference=False)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Skip connections: {model.use_skip_connections}")
    print(f"Output projection: {model.output_projection}")
    print("✓ Skip connections work!\n")


def example_tcn_with_embeddings():
    """Example 3: TCN with embeddings."""
    print("="*70)
    print("Example 3: TCN with Embeddings (concat mode)")
    print("="*70)
    
    model = TCN(
        num_inputs=8,
        num_channels=[16, 32],
        kernel_sizes=3,
        embedding_shapes=[(12,)],
        embedding_mode="concat",
        causal=False,
    )
    
    batch, seq_len = 2, 20
    x = mx.random.normal((batch, seq_len, 8))
    embeddings = [mx.random.normal((batch, 12))]  # Will be broadcasted across time
    
    output = model(x, embeddings=embeddings, inference=False)
    
    print(f"Input shape:      {x.shape}")
    print(f"Embedding shape:  {embeddings[0].shape}")
    print(f"Output shape:     {output.shape}")
    print(f"Embedding mode:   {model.embedding_mode}")
    print("✓ Embeddings work!\n")


def example_causal_streaming():
    """Example 4: Causal TCN with streaming inference."""
    print("="*70)
    print("Example 4: Causal TCN with Streaming Inference")
    print("="*70)
    
    model = TCN(
        num_inputs=4,
        num_channels=[8, 16],
        kernel_sizes=3,
        causal=True,
        dropout=0.0,
    )
    
    # Offline mode (full sequence)
    batch, full_seq_len = 1, 50
    x_full = mx.random.normal((batch, full_seq_len, 4))
    output_offline = model(x_full, embeddings=None, inference=False)
    
    print(f"Offline mode:")
    print(f"  Input shape:  {x_full.shape}")
    print(f"  Output shape: {output_offline.shape}")
    
    # Streaming mode (chunk-by-chunk)
    model.reset_buffers()
    chunk_size = 10
    num_chunks = 5
    
    outputs_streaming = []
    for i in range(num_chunks):
        chunk = mx.random.normal((1, chunk_size, 4))
        out_chunk = model(chunk, embeddings=None, inference=True)
        outputs_streaming.append(out_chunk)
    
    print(f"\nStreaming mode ({num_chunks} chunks):")
    print(f"  Chunk size: {chunk_size}")
    for i, out in enumerate(outputs_streaming):
        print(f"  Chunk {i+1} output shape: {out.shape}")
    
    print("✓ Streaming inference works!\n")


def example_variable_kernel_sizes():
    """Example 5: TCN with different kernel sizes per layer."""
    print("="*70)
    print("Example 5: TCN with Variable Kernel Sizes")
    print("="*70)
    
    model = TCN(
        num_inputs=8,
        num_channels=[16, 32, 64],
        kernel_sizes=[3, 5, 7],  # Different kernel size per layer
        causal=False,
    )
    
    batch, seq_len = 2, 50
    x = mx.random.normal((batch, seq_len, 8))
    output = model(x, embeddings=None, inference=False)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Kernel sizes: [3, 5, 7] (different per layer)")
    print(f"Number of layers: {len(model.network)}")
    for idx, block in enumerate(model.network):
        # MLX Conv1d weight shape: (out_channels, kernel_size, in_channels)
        ks = block.conv1.weight.shape[1]
        print(f"  Layer {idx+1} kernel size: {ks}")
    print("✓ Variable kernel sizes work!\n")


def example_all_features():
    """Example 6: TCN with all features enabled."""
    print("="*70)
    print("Example 6: TCN with All Features")
    print("="*70)
    
    model = TCN(
        num_inputs=8,
        num_channels=[16, 32, 64],
        kernel_sizes=3,
        dilations=[1, 2, 4],
        dropout=0.1,
        causal=True,
        use_norm="layer_norm",
        activation="gelu",
        kernel_initilaizer="xavier_normal",  # xavier works with gelu
        use_skip_connections=True,
        embedding_shapes=[(16,)],
        embedding_mode="add",
        use_gate=True,
        output_projection=10,
        output_activation="sigmoid",
    )
    
    batch, seq_len = 1, 30
    x = mx.random.normal((batch, seq_len, 8))
    embeddings = [mx.random.normal((batch, 16))]
    
    output = model(x, embeddings=embeddings, inference=False)
    
    print(f"Features enabled:")
    print(f"  - Skip connections: {model.use_skip_connections}")
    print(f"  - Gated activation: {model.use_gate}")
    print(f"  - Embeddings: {model.embedding_shapes} ({model.embedding_mode} mode)")
    print(f"  - Output projection: {model.output_projection}")
    print(f"  - Output activation: {model.output_activation}")
    print(f"  - Custom dilations: {model.dilations}")
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ All features work together!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MLX-TCN Examples")
    print("="*70 + "\n")
    
    try:
        example_basic_tcn()
        example_tcn_with_skip_connections()
        example_tcn_with_embeddings()
        example_causal_streaming()
        example_variable_kernel_sizes()
        example_all_features()
        
        print("="*70)
        print("All examples completed successfully! ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
