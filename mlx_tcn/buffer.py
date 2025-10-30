"""Utilities for managing streaming buffers shared across padded modules."""

import mlx.core as mx
from typing import Optional
from collections.abc import Iterable

class BufferIO(Iterable):
    """Helper object that tracks incoming, outgoing, and internal buffers across steps."""
    def __init__(self, in_buffers: Optional[Iterable] = None):
        """Optionally seed the iterator with precomputed input buffers."""
        if in_buffers is None:
            self.in_buffer = None
            self.in_buffer_length = 0
        else:
            buffers = list(in_buffers)
            self.in_buffer = iter(buffers)
            self.in_buffer_length = len(buffers)

        self.out_buffers = []
        self.internal_buffer = []

    def __iter__(self) -> any:
        """Return iterator handle so BufferIO can be consumed in loops."""
        return self  

    def __next__(self) -> any:
        """Yield the next input buffer or return None when exhausted."""
        if self.in_buffer is not None:
            try:
                return next(self.in_buffer)
            except StopIteration:
                return None
        else:
            return None

    def append_out_buffer(self, x: mx.array):
        """Store an output buffer produced during streaming."""
        self.out_buffers.append(x)

    def append_internal_buffer(self, x: mx.array):
        """Persist an internally generated buffer for later reuse."""
        self.internal_buffer.append(x)

    def next_in_buffer(self) -> any:
        """Advance the iterator to the next input buffer."""
        return self.__next__()
        
    def step(self):
        """Move buffered outputs into the input iterator for the next inference step."""
        if self.in_buffer is None:
            self.in_buffer_length = len(self.internal_buffer)
            self.in_buffer = iter(self.internal_buffer)
            self.internal_buffer = []
        if len(self.out_buffers) != self.in_buffer_length:
            raise ValueError(f"""The number of out buffers must be equal to the number of in buffers, but got {len(self.out_buffers)} and {self.in_buffer_length}""")
        self.in_buffer_length = len(self.out_buffers)
        self.in_buffer = iter(self.out_buffers)
        self.out_buffers = []
