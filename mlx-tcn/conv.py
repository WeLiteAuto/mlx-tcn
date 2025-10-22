"""Temporal convolution layers that wrap MLX ops with streaming-aware padding."""

import mlx.core as mx
import mlx.nn as nn
from typing import Union, Optional

from pad import TemporalPad1d
from buffer import BufferIO

class TemporalConv1d(nn.Conv1d):
    """Conv1d variant that auto-computes padding and maintains a temporal buffer."""
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride: int, 
                padding: int = 0, 
                dilation: int = 1, 
                groups: int = 1, 
                bias: bool = True, 
                padding_mode: str = "zeros",
                buffer: Optional[Union[int,float, mx.array]] = None,
                causal: bool = False,
                look_ahead: int = 0):
        """Initialize the layer and hook up a TemporalPad1d helper."""
        if padding != 0:
            raise ValueError(f"""
             The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
            """)
        
        if look_ahead != 0:
            raise ValueError(f"""
            The value of arg 'look_ahead' must be 0 for TemporalConv1d, because the correct amount
            of look_ahead is calculated automatically based on the kernel size and dilation.
            The value of 'look_ahead' will be ignored.
            """)

        super(TemporalConv1d, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride,
                                            0,
                                            dilation,
                                            groups,
                                            bias)
        
        
        self.pad_len = (kernel_size - 1) * dilation
        self.causal = causal

        self.padder = TemporalPad1d(padding=self.pad_len, 
                                    in_channels=in_channels,
                                    buffer=buffer, 
                                    padding_mode=padding_mode,
                                    causal=self.causal)
    
    def reset_buffer(self):
        """Expose padder reset so callers can clear streaming state."""
        self.padder.reset_buffer()

    def __call__(self, x: mx.array, inference: bool = False, buffer_io: Optional[BufferIO] = None) -> mx.array:
        """Apply temporal padding then forward through the underlying convolution."""
        out = self.padder(x, inference, buffer_io)
        out = super(TemporalConv1d, self).__call__(out)
        return out
      
        	
      
class TemporalConTransposed1d(nn.ConvTransposed1d):
    """ConvTranspose1d analogue with controlled buffering for causal decoding."""
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride: int, 
                padding: int, 
                output_padding: int = 0,
                groups: int = 1,
                bias: bool = True,
                dilation: int=1, 
                padding_mode: str = "zeros",
                buffer: Optional[Union[int,float, mx.array]] = None,
                causal: bool = False,
                look_ahead: int = 0):
        """Validate arguments and prepare TemporalPad1d-based buffering."""
    
        if padding != 0:
            raise ValueError(f"""
             The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
            """)

        if dilation != 1:
            raise ValueError(f"""
            The value of arg 'dilation' must be 1 for TemporalConv1d, because the correct amount
            of dilation is calculated automatically based on the kernel size and stride.
            The value of 'dilation' will be ignored.
            """)
        
        if look_ahead != 0:
            raise ValueError(f"""
            The value of arg 'look_ahead' must be 0 for TemporalConv1d, because the correct amount
            of look_ahead is calculated automatically based on the kernel size and dilation.
            The value of 'look_ahead' will be ignored.
            """)

        if output_padding != 0:
            raise ValueError(f"""
            The value of arg 'output_padding' must be 0 for TemporalConv1d, because the correct amount
            of output padding is calculated automatically based on the kernel size and stride.
            The value of 'output_padding' will be ignored.
            """)
        
        if kernel_size != 2 * stride:
            raise ValueError(f"""
                This implementation of TemporalConvTranspose1d only
                supports kernel_size == 2 * stride, but got 
                kernel_size = {kernel_size} and stride = {stride}.
                """)

        self.causal = causal
        self.upsampling_factor = stride
        self.buffer_size = (kernel_size // stride) - 1

        if self.causal:
            self.implicit_padding = 0
        else:
            self.implicit_padding = (kernel_size - stride) // 2

        

        super(TemporalConTransposed1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, output_padding)
        
        self.padder = TemporalPad1d(padding=self.buffer_size,
                                    in_channels=in_channels,
                                    padding_mode=padding_mode,
                                    causal=self.causal)

        if buffer is None:
            buffer = mx.zeros((1, in_channels, self.buffer_size))
        self.buffer = buffer

    def reset_buffer(self):
        """Delegate buffer reset to the internal padder."""
        self.padder.reset_buffer()

    def __call__(self, x: mx.array, inference: bool = False, buffer_io: Optional[BufferIO] = None) -> mx.array:
        """Perform causal or non-causal transpose convolution with buffering."""
        if self.causal:
            out = self.padder(x, inference, buffer_io)
            out = super(x)
            out = out[..., self.upsampling_factor: - self.upsampling_factor]
        else:
            out = super(x)
            if self.upsampling_factor % 2 == 1:
                out = out[..., :-1]
        return out
