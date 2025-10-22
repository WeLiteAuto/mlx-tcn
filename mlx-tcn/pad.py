import mlx.core as mx
import mlx.nn as nn
from typing import Union, Optional
from buffer import BufferIO


PADDING_MODES = [
    "reflect",
    "replicate",
    "circular",
    "zeros",
]

class TemporalPad1d(nn.Module):
    """Temporal padding utility that supports training-time padding and streaming inference."""
    def __init__(self, padding: int, in_channels: int, buffer: Optional[Union[float, mx.array]], padding_mode: str = "zeros", causal: bool = False):
        """Configure padding lengths, mode, and initialize the temporal buffer."""
        super(TemporalPad1d, self).__init__()
       
        self.pad_len = padding
        self.causal = causal
        
        if causal:
            self.left_padding = self.pad_len
            self.right_padding = 0
        else:
            self.left_padding = self.pad_len // 2
            self.right_padding = self.pad_len - self.left_padding
        
        if padding_mode not in PADDING_MODES:
            raise ValueError(f"""padding_mode must be one of {PADDING_MODES}, but got {padding_mode}""")
        self.padding_mode = padding_mode


        # 初始化历史帧缓冲区，用于推理阶段拼接上下文
        if buffer is None:
            self.buffer = mx.zeros((1, self.pad_len, in_channels))
        elif isinstance(buffer, (int, float)):
            self.buffer = mx.full((1, self.pad_len, in_channels), buffer)
        elif not isinstance(buffer, mx.array):
            raise ValueError(f"""buffer must be a int, float or mx.array, but got {type(buffer)}""")
        else:
            self.buffer = buffer


    def pad_inference(self, x: mx.array, buffer_io: Optional[BufferIO] = None) -> mx.array:
        """Apply causal padding for streaming inference while maintaining an external or internal buffer."""
        if not self.causal:
            raise ValueError(f"""
            Causal padding is not supported for inference
            """)

        if x.shape[0] != 1:
            raise ValueError(f"""
                Streaming inference requires a batch size
                of 1, but batch size is {x.shape[0]}.
                """)
        if buffer_io is None:
            in_buffer = self.buffer
        else:
            in_buffer = buffer_io.next_in_buffer()
            if in_buffer is None:
                in_buffer = self.buffer
                buffer_io.append_internal_buffer(in_buffer)
        # 把历史帧拼到当前输入左侧，形成 (context + current chunk)
        x = mx.concatenate((in_buffer, x), axis=1)
        out_buffer = x[:, -self.pad_len:, :]
        if buffer_io is  None:
            self.buffer = out_buffer
        else:
            buffer_io.append_out_buffer(out_buffer)

        return x

    def reset_buffer(self):
        """Zero out the internal buffer and validate its length."""
        # 推理过程中可通过该方法将缓冲区清零
        self.buffer = mx.zeros_like(self.buffer)
        if self.buffer.shape[1] != self.pad_len:
            raise ValueError(f"""
            Buffer length must be {self.pad_len}, but got {self.buffer.shape[1]}
            """)

    def __call__(self,
                 x: mx.array, 
                inference: bool = False, 
                buffer_io: Optional[BufferIO] = None) -> mx.array:
        """Dispatch to streaming or offline padding depending on the inference flag."""
        
        if inference:
            x = self.pad_inference(x, buffer_io)
        else:
            if self.padding_mode == "zeros":
                x = mx.pad(x, ((0, 0), (self.left_padding, self.right_padding), (0, 0)), mode="constant", constant_values=0)
            elif self.padding_mode == "reflect":
                x = mx.pad(x, ((0, 0), (self.left_padding, self.right_padding), (0, 0)), mode="reflect")
            elif self.padding_mode == "replicate":
                x = mx.pad(x, ((0, 0), (self.left_padding, self.right_padding), (0, 0)), mode="edge")
            elif self.padding_mode == "circular":
                x = mx.pad(x, ((0, 0), (self.left_padding, self.right_padding), (0, 0)), mode="wrap")
            else:
                raise ValueError(f"""Invalid padding mode: {self.padding_mode}""")

        return x

                
    
       
      

       

       
        
        
        
        
 
