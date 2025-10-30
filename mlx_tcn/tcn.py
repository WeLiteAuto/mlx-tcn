import math
from collections.abc import Iterable
from typing import Callable, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn

from .buffer import BufferIO
from .conv import TemporalConv1d, TemporalConvTransposed1d
from .pad import TemporalPad1d
from .init import calculate_gain
from .parametrizations import weight_norm


activation_fn = dict(
    relu=nn.ReLU,
    leaky_relu=nn.LeakyReLU,
    gelu=nn.GELU,
    elu=nn.ELU,
    selu=nn.SELU,
    tanh=nn.Tanh,
    sigmoid=nn.Sigmoid,
    softmax=nn.Softmax,
    log_softmax=nn.LogSoftmax,
)

kernel_init_fn = dict(
    he_normal=nn.init.he_normal,
    he_uniform=nn.init.he_uniform,
    xavier_normal=nn.init.glorot_normal,
    xavier_uniform=nn.init.glorot_uniform,
    normal=nn.init.normal,
    uniform=nn.init.uniform,
)


def _infer_activation_key(activation: Union[str, Type[nn.Module]]) -> Optional[str]:
    """Return the canonical activation key if known."""
    if isinstance(activation, str):
        key = activation.lower()
        return key if key in activation_fn else None

    activation_cls = activation if isinstance(activation, type) else type(activation)

    for key, candidate in activation_fn.items():
        if activation_cls is candidate or issubclass(activation_cls, candidate):
            return key
    return None


def get_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_kernel_init_fn(name: str, activation: Optional[str]) -> Tuple[Callable, dict]:
    if name not in kernel_init_fn:
        raise ValueError(f"Invalid kernel init: {name}")

    init_fn = kernel_init_fn[name]
    activation_key = activation.lower() if activation is not None else None

    if name in ["xavier_normal", "xavier_uniform"]:
        if activation_key in {"gelu", "elu", "softmax", "log_softmax"}:
            gain = math.sqrt(2)
        else:
            try:
                gain = calculate_gain(activation_key) if activation_key else 1.0
            except ValueError:
                gain = 1.0

        kernel_init_kw = {"gain": gain}
    elif name in ["he_normal", "he_uniform"]:
        if activation_key in {"gelu", "elu", "softmax", "log_softmax"}:
            raise ValueError(
                f"Invalid activation: {activation} for kernel init {name}, "
                "use 'relu' or 'leaky_relu' instead"
            )

        try:
            gain = calculate_gain(activation_key) if activation_key else 1.0
        except ValueError:
            gain = 1.0

        kernel_init_kw = {"gain": gain}
    else:
        kernel_init_kw = {}
    return init_fn, kernel_init_kw



class BaseTCN(nn.Module):
    """Utility base class for temporal convolutional networks with streaming helpers."""
    def __init__(self):
        """Initialize shared TCN state; subclasses must implement the forward pass."""
        super(BaseTCN, self).__init__()

    def inference(self, *arg, **kwargs) -> mx.array:
        """Run the model in inference mode by forcing `inference=True`."""
        return self(*arg, inference=True, **kwargs)

    def init_weights(self):
        """Reinitialize convolution weights with a small normal distribution."""
        def _init_weight(name: str, m: nn.Module):    
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight = nn.init.normal(mean=0.0, std=0.01)(m.weight)
        self.apply_to_modules(_init_weight)

    def reset_buffers(self):
        """Clear all TemporalPad1d buffers to their default state."""
        def _reset_buffers(name: str, m: nn.Module):
            if isinstance(m, (TemporalPad1d)):
                m.reset_buffer()
        self.apply_to_modules(_reset_buffers)

    def get_buffers(self) -> List[mx.array]:
        """Collect TemporalPad1d buffers in creation order."""
        buffers = []
        def _get_buffers(name: str, m: nn.Module):
            if isinstance(m, (TemporalPad1d,)):
                buffers.append(m.buffer)
        self.apply_to_modules(_get_buffers)
        return buffers


    def get_in_buffers(self, *args, **kwargs) -> List[mx.array]:
        """Return buffers ordered by usage during a streaming forward pass."""
        buffers = self.get_buffers()
        buffer_io = BufferIO(in_buffers=None)
        self(*args, inference=True, in_buffer=buffer_io, **kwargs)
        in_buffers = buffer_io.internal_buffer
        self.set_buffers(buffers)
        return in_buffers


    def set_buffers(self, buffers: List[mx.array]):
        """Restore TemporalPad1d buffers using the provided list."""
        def _set_buffers(name: str, m: nn.Module):
            if isinstance(m, (TemporalPad1d,)):
                m.buffer = buffers.pop(0)
        self.apply_to_modules(_set_buffers)


class TemporalBlock(BaseTCN):
    def __init__(self, 
                in_channels : int,
                out_channels : int,
                kernel_size : int,
                stride : int,
                dilation : int,
                dropout: float,
                causal: bool,
                norm: str,
                activation: Union[str, Type[nn.Module]],
                kernel_init: str,
                embedding_dims: Optional[List[mx.array]],
                embedding_mode: str = "add",
                use_gate: bool=False):
        

        super(TemporalBlock, self).__init__()
        self.norm = norm
        self.activation = activation
        self.kernel_init = kernel_init
        self.use_gate = use_gate
        self.causal = causal
        object.__setattr__(self, "_embedding_dims", None)

        if self.use_gate:
            conv1d_n_outputs = 2 * out_channels
        else:
            conv1d_n_outputs = out_channels

        self.conv1 = TemporalConv1d(in_channels=in_channels,
                                    out_channels=conv1d_n_outputs,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    causal=self.causal)

        self.conv2 = TemporalConv1d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    causal=self.causal)

        if norm == "batch_norm":
            if self.use_gate:
                self.norm1 = nn.BatchNorm(num_features=2 * out_channels)
            else:
                self.norm1 = nn.BatchNorm(num_features=out_channels)
            self.norm2 = nn.BatchNorm(num_features=out_channels)
        elif norm == "layer_norm":
            if self.use_gate:
                self.norm1 = nn.LayerNorm(dims=2 * out_channels)
            else:
                self.norm1 = nn.LayerNorm(dims=out_channels)
            self.norm2 = nn.LayerNorm(dims=out_channels)
        elif norm == "weight_norm":
            self.norm1 = None
            self.norm2 = None
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
        elif norm is None:
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"Invalid norm: {norm}")


        activation_key: Optional[str] = None
        if isinstance(activation, str):
            activation_key = activation.lower()
            act_cls = activation_fn.get(activation_key)
            if act_cls is None:
                raise ValueError(f"Invalid activation: {activation}")
            self.activation1 = act_cls()
            self.activation2 = act_cls()
            self.activation_final = act_cls()
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            self.activation1 = activation()
            self.activation2 = activation()
            self.activation_final = activation()
            activation_key = _infer_activation_key(activation)
        else:
            raise ValueError(f"Invalid activation: {activation}")

        for module in (self.activation1, self.activation2, self.activation_final):
            if not isinstance(module, nn.Module):
                raise TypeError("Activation factory must produce nn.Module instances.")

        self.activation_name = activation_key


        if self.use_gate:
            self.activation1 = nn.GLU(axis=-1)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.downSample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else None
        
        if embedding_dims is not None:
            sanitized_dims = []
            for shape in embedding_dims:
                if not isinstance(shape, (tuple, list)):
                    raise ValueError("Embedding dims must be tuple/list of integers.")
                sanitized_dims.append(tuple(shape))
            object.__setattr__(self, "_embedding_dims", tuple(sanitized_dims))

        if self.embedding_dims is not None:
            # Embedding projections support both 'concat' and 'add' modes
            if self.use_gate:
                embedding_layer_n_outputs = 2 * out_channels
            else:
                embedding_layer_n_outputs = out_channels
            
            embedding_total_in = sum(shape[0] for shape in self.embedding_dims)
            self.embedding_projection1 = nn.Conv1d(
                in_channels=embedding_total_in,
                out_channels=embedding_layer_n_outputs,
                kernel_size=1,
            )
            self.embedding_projection2 = nn.Conv1d(
                in_channels=2 * embedding_layer_n_outputs,
                out_channels=embedding_layer_n_outputs,
                kernel_size=1,
            )
        self.embedding_mode = embedding_mode
        
        self.init_weights()
    

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_init, activation=self.activation_name
        )
        # MLX init functions return a callable, not accept kwargs directly
        init_fn = initialize()
        self.conv1.weight = init_fn(self.conv1.weight)
        self.conv2.weight = init_fn(self.conv2.weight)
        if self.downSample is not None:
            self.downSample.weight = init_fn(self.downSample.weight)
        embedding_proj_1 = getattr(self, "embedding_projection1", None)
        if embedding_proj_1 is not None:
            embedding_proj_1.weight = init_fn(embedding_proj_1.weight)
        embedding_proj_2 = getattr(self, "embedding_projection2", None)
        if embedding_proj_2 is not None:
            embedding_proj_2.weight = init_fn(embedding_proj_2.weight)

    def apply_normal(self, norm_fn: Callable, x: mx.array) -> mx.array:
        return norm_fn(x) if norm_fn is not None else x

    def apply_embedding(self, x: mx.array, embeddings: Union[List[mx.array], mx.array]):
        if self.embedding_dims is None:
            raise ValueError("Embeddings were provided, but this block was not configured with embedding_dims.")

        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        if len(embeddings) != len(self.embedding_dims):
            raise ValueError(
                f"Expected {len(self.embedding_dims)} embeddings, but received {len(embeddings)}."
            )

        enriched: List[mx.array] = []
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        for embedding, expected_shape in zip(embeddings, self.embedding_dims):
            expected_dim = expected_shape[0]

            if embedding.shape[0] != batch_size:
                raise ValueError(
                    f"Embedding batch mismatch: expected {batch_size}, got {embedding.shape[0]}."
                )

            if len(embedding.shape) == 2:
                if embedding.shape[1] != expected_dim:
                    raise ValueError(
                        f"Embedding feature mismatch: expected {expected_dim}, got {embedding.shape[1]}."
                    )
                emb_time = mx.expand_dims(embedding, axis=1)
                emb_time = mx.broadcast_to(emb_time, (batch_size, seq_len, expected_dim))
                enriched.append(emb_time)
            elif len(embedding.shape) == 3:
                if embedding.shape[1] != seq_len:
                    raise ValueError(
                        f"Embedding length mismatch: expected {seq_len}, got {embedding.shape[1]}."
                    )
                if embedding.shape[2] != expected_dim:
                    raise ValueError(
                        f"Embedding feature mismatch: expected {expected_dim}, got {embedding.shape[2]}."
                    )
                enriched.append(embedding)
            else:
                raise ValueError(
                    f"Unsupported embedding rank {len(embedding.shape)}; expected 2 or 3 dimensions."
                )

        combined = mx.concatenate(enriched, axis=-1)
        combined = self.embedding_projection1(combined)

        if self.embedding_mode == 'concat':
            out = self.embedding_projection2(mx.concatenate([x, combined], axis=-1))
        elif self.embedding_mode == 'add':
            out = x + combined
        else:
            raise ValueError(f"Invalid embedding_mode: {self.embedding_mode}. Must be 'add' or 'concat'.")
        return out

    def __call__(self, x: mx.array, 
                embeddings: Optional[Union[List[mx.array], mx.array]] = None, 
                inference: bool = False, 
                in_buffer: Optional[List[BufferIO]] = None) -> Tuple[mx.array, mx.array]:
        if in_buffer is not None:
            in_buffer_1, in_buffer_2 = in_buffer
        else:
            in_buffer_1 = None
            in_buffer_2 = None

        out = self.conv1(x, inference=inference, buffer_io=in_buffer_1)
        out = self.apply_normal(self.norm1, out)
        if embeddings is not None:
            out = self.apply_embedding(out, embeddings)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out, inference=inference, buffer_io=in_buffer_2)
        out = self.apply_normal(self.norm2, out)
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downSample is None else self.downSample(x)
        return self.activation_final(res + out), out

    @property
    def embedding_dims(self):
        return getattr(self, "_embedding_dims", None)



class TCN(BaseTCN):
    def __init__(self,
                num_inputs: int,
                num_channels: Union[List[int], mx.array],
                kernel_sizes: Union[List[int], int] = 4,
                dilations: Optional[ Union[List[int], mx.array]] = None,
                dilation_reset: Optional[int] = None,
                dropout: float = 0.1,
                causal: bool = False,
                use_norm: str = "batch_norm",
                activation: str = "relu",
                kernel_initilaizer: str = "he_normal",
                use_skip_connections: bool = False,
                embedding_shapes: Optional[List[Tuple]] = None,
                embedding_mode: str = "add",
                use_gate: bool = False,
                look_ahead: int = 0,
                output_projection: Optional[int] = None,
                output_activation: Optional[str] = None):
        
        super(TCN, self).__init__()

        if look_ahead != 0:
            raise ValueError(f"""The value of arg 'look_ahead' must be 0 for TCN, because the correct amount
            of look_ahead is calculated automatically based on the kernel size and dilation.
            The value of 'look_ahead' will be ignored.
            """)
        
        if dilations is not None:
            if len(dilations) != len(num_channels):
                raise ValueError(f"Length of dilations must match the length of num_channels.")
        else:
            if dilation_reset is None:
                dilations = [2 ** ii for ii in range(len(num_channels))]
            else :
                dilation_reset = int(mx.log2(dilation_reset * 2))
                dilations = [2 ** (ii % dilation_reset) for ii in range(len(num_channels))]
        
        object.__setattr__(self, "_dilations", tuple(dilations))
        self.activation = activation
        self.kernel_init = kernel_initilaizer
        self.use_skip_connections = use_skip_connections
        object.__setattr__(self, "_embedding_shapes", None)
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.output_projection = output_projection
        self.output_activation = output_activation


        if embedding_shapes is not None:
            if isinstance(embedding_shapes, Iterable):
                sanitized_shapes = []
                for shape in embedding_shapes:
                    if not isinstance(shape, tuple):
                        try:
                            shape = tuple(shape)
                        except (TypeError, ValueError):
                            raise ValueError(f"Invalid embedding shape: {shape}. Must be a tuple of integers.")

                    if len(shape) not in [1, 2]:
                        raise ValueError(f"Invalid embedding shape: {shape}. Must be a tuple of one or two integers.")
                    sanitized_shapes.append(shape)
            else:
                raise ValueError(f"Invalid embedding shapes: {embedding_shapes}. Must be an iterable of tuples of one or two integers.")
            object.__setattr__(self, "_embedding_shapes", tuple(sanitized_shapes))
        else:
            object.__setattr__(self, "_embedding_shapes", None)
        
        if use_skip_connections:
            self.downsample_skip_connection : List[Optional[nn.Module]] = []
            for ii in range(len(num_channels)):
                if num_channels[ii] != num_channels[-1]:
                    self.downsample_skip_connection.append(
                        nn.Conv1d(in_channels=num_channels[ii], out_channels=num_channels[-1], kernel_size=1)
                    )
                else:
                    self.downsample_skip_connection.append(None)
            
            self.init_skip_connections_weights()
            if isinstance(self.activation, str):
                activation_key = self.activation.lower()
                act_cls = activation_fn.get(activation_key)
                if act_cls is None:
                    raise ValueError(f"Invalid activation: {self.activation}")
                self.activation_skip_out = act_cls()
            elif isinstance(self.activation, type) and issubclass(self.activation, nn.Module):
                self.activation_skip_out = self.activation()
            else:
                raise ValueError(f"Invalid activation: {self.activation}")
        else:
            self.downsample_skip_connection = None

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(num_channels)

        self.network : List[TemporalBlock] = []
        for ii in range(len(num_channels)):
            dilation = self.dilations[ii]
            in_channels = num_inputs if ii == 0 else num_channels[ii - 1]
            out_channels = num_channels[ii]
            kernel_size = kernel_sizes[ii]
            self.network.append(TemporalBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              dilation=dilation,
                                              dropout=dropout,
                                              causal=self.causal,
                                              norm=use_norm,
                                              activation=self.activation,
                                              kernel_init=self.kernel_init,
                                              embedding_dims=self.embedding_shapes,
                                              embedding_mode=self.embedding_mode,
                                              use_gate=self.use_gate))
        
        if self.output_projection is not None:
            self.projection_out = nn.Conv1d(in_channels=num_channels[-1], 
                                out_channels=self.output_projection, 
                                kernel_size=1)
        else:
            self.projection_out = None

        if self.output_activation is not None:
            if isinstance(self.output_activation, str):
                activation_key = self.output_activation.lower()
                activation_cls = activation_fn.get(activation_key)
                if activation_cls is None:
                    raise ValueError(f"Invalid activation: {self.output_activation}")
                self.activation_out = activation_cls()
            elif isinstance(self.output_activation, type) and issubclass(self.output_activation, nn.Module):
                self.activation_out = self.output_activation()
            else:
                raise ValueError(f"Invalid activation: {self.output_activation}")
        else:
            self.activation_out = None
        
        if self.causal:
            self.reset_buffers()

    @property
    def dilations(self):
        return getattr(self, "_dilations", tuple())

    @property
    def embedding_shapes(self):
        return getattr(self, "_embedding_shapes", None)


    def init_skip_connections_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_init, activation=self.activation
        )

        init_fn = initialize()
        for layer in self.downsample_skip_connection:
            if layer is not None:
                layer.weight = init_fn(layer.weight)

    def __call__(self, x: mx.array, 
                embeddings: Optional[Union[List[mx.array], mx.array]] = None, 
                inference: bool = False, 
                in_buffer: Optional[List[BufferIO]] = None) -> mx.array:
       
        if inference and (not self.causal):
            raise ValueError(f"""
                Streaming inference is only supported for causal TCNs.
                Expected inference=True when causal=True, but got inference={inference} and causal={self.causal}.
                """)
        if self.use_skip_connections:
            skip_connections : List[mx.array] = []
            for index, layer in enumerate(self.network):
                if in_buffer is not None:
                    layer_in_buffer = in_buffer[index*2:index*2+2]
                else:
                    layer_in_buffer = None
                
                x, skip_out = layer(x, embeddings, inference, layer_in_buffer)

                if self.downsample_skip_connection[index] is not None:
                    skip_out = self.downsample_skip_connection[index](skip_out)
                if index < len(self.network) - 1:
                    skip_connections.append(skip_out)
            skip_connections.append(x)
            x = mx.stack(skip_connections, axis=0).sum(axis=0)
            x = self.activation_skip_out(x)
        else:
            for index, layer in enumerate(self.network):
                if in_buffer is not None:
                    layer_in_buffer = in_buffer[index*2:index*2+2]
                else:
                    layer_in_buffer = None
                
                x, _ = layer(x, embeddings, inference, layer_in_buffer)
        
        if self.projection_out is not None:
            x = self.projection_out(x)
        if self.activation_out is not None:
            x = self.activation_out(x)
        return x
