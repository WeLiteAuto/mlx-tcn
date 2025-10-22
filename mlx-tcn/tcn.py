import mlx.core as mx
import mlx.nn as nn 
from typing import Optional, Union, Tuple, List, Callable, Type
from collections.abc import Iterable
from mlx.nn.layers.normalization import InstanceNorm
from numpy import isin
from conv import TemporalConv1d, TemporalConTransposed1d
from pad import TemporalPad1d
from buffer import BufferIO
from utils import calculate_gain
import math


activation_fn = dict(
    relu=nn.ReLU,
    leaky_relu=nn.LeakyReLU,
    gelu=nn.GELU,
    elu=nn.ELU,
    silu=nn.SiLU,
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

    def init_weight(self):
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
        self(*args, inference=True, buffer_io=buffer_io, **kwargs)
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
                embedding_mode: str,
                use_gate: bool=False):
        

        super(TemporalBlock, self).__init__()
        self.norm = norm
        self.activation = activation
        self.kernel_init = kernel_init
        self.embedding_dims = embedding_dims
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal

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
                self.norm1 = nn.LayerNorm(normalized_shape=2 * out_channels)
            else:
                self.norm1 = nn.LayerNorm(normalized_shape=out_channels)
            self.norm2 = nn.LayerNorm(normalized_shape=out_channels)
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
        
        if self.embedding_dims is not None:
            if self.use_gate:
                embedding_layer_n_outputs = 2 * out_channels
            else:
                embedding_layer_n_outputs = out_channels
            
            self.embedding_projection1 = \
                nn.Conv1d(in_channels= sum([ shape[0] for shape in embedding_dims]),
                        out_channels=embedding_layer_n_outputs, 
                        kernel_size=1)
            self.embedding_projection2 = \
                nn.Conv1d(in_channels= 2 * embedding_layer_n_outputs,
                        out_channels=embedding_layer_n_outputs, 
                        kernel_size=1)
        
        self.init_weights()
    

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_init, activation=self.activation_name
        )
        initialize(self.conv1.weight, **kwargs)
        initialize(self.conv2.weight, **kwargs)
        if self.downSample is not None:
            initialize(self.downSample.weight, **kwargs)

    def apply_normal(self, norm_fn: Callable, x: mx.array) -> mx.array:
        return norm_fn(x) if norm_fn is not None else x

    def apply_embedding(self, x: mx.array, embeddings: Union[List[mx.array], mx.array]):
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
       
        
        e = []

        for embedding , expected_shape in zip(embeddings, self.embedding_dims):
            if embedding.shape[1] != expected_shape[0]:
                raise ValueError(f"""
                Expected embedding shape {expected_shape[0]}, 
                but got {embedding.shape[1]}
                """)
            if len(embedding.shape) == 2:

                emb_time = mx.expand_dims(embedding, axis=1)
                emb_time = mx.broadcast_to(emb_time, (embedding.shape[0], x.shape[1], embedding.shape[1]))
                e.append(emb_time)
            elif len(embedding.shape) == 3:
                if embedding.shape[1] != x.shape[1]:
                    raise ValueError(f"""
                    Expected embedding shape {x.shape[1]}, 
                    but got {embedding.shape[1]}
                    """)
                e.append(embedding)

        e = mx.concatenate(e, axis=-1)
        e = self.embedding_projection1(e)
        out = self.embedding_projection2(mx.concatenate([x, e], axis=-1))
        return out

    def __call__(self, x: mx.array, 
                embeddings: Union[List[mx.array], mx.array], 
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