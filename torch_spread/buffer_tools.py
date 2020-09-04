from typing import Union, Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .buffer import Buffer
from .utilities import ShapeBufferType, DtypeBufferType, BufferType


def buffer_fill_information(buffer: BufferType,
                            shape: Optional[ShapeBufferType] = None,
                            dtype: Optional[DtypeBufferType] = None,
                            size: Optional[int] = None) -> Tuple[ShapeBufferType, DtypeBufferType, int]:
    if shape is None:
        shape = buffer_map(lambda x: x.shape[1:], buffer)

    if dtype is None:
        dtype = buffer_map(lambda x: x.dtype, buffer)

    if size is None:
        size = buffer_size(buffer)

    return shape, dtype, size


def iterate_buffer(buffer: BufferType, context: tuple = tuple()):
    if isinstance(buffer, dict):
        for key, value in buffer.items():
            yield from iterate_buffer(value, context + (key,))
    elif isinstance(buffer, list):
        for key, value in enumerate(buffer):
            yield from iterate_buffer(value, context + (key,))
    else:
        if len(context) == 1:
            yield context[0], buffer
        else:
            yield context, buffer


def buffer_map(method: Callable[[Tensor], Tensor], buffer: BufferType):
    if isinstance(buffer, dict):
        return {key: buffer_map(method, value) for key, value in buffer.items()}
    elif isinstance(buffer, list):
        return [buffer_map(method, value) for value in buffer]
    else:
        return method(buffer)


def buffer_multi_map(method: Callable, buffer: BufferType, *buffers: BufferType):
    if isinstance(buffer, dict):
        return {key: buffer_multi_map(method, value, *(buff[key] for buff in buffers)) for key, value in buffer.items()}
    elif isinstance(buffer, list):
        return [buffer_multi_map(method, *values) for values in zip(buffer, *buffers)]
    else:
        return method(buffer, *buffers)


def buffer_map_reduce(method: Callable[[Tensor], Tensor], reduction: Callable, buffer: BufferType):
    if isinstance(buffer, dict):
        return reduction(buffer_map_reduce(method, reduction, value) for value in buffer.values())
    elif isinstance(buffer, list):
        return reduction(buffer_map_reduce(method, reduction, value) for value in buffer)
    else:
        return method(buffer)


def buffer_multi_map_reduce(method: Callable, reduction: Callable, buffer: BufferType, *buffers: BufferType):
    if isinstance(buffer, dict):
        return reduction(
            buffer_multi_map_reduce(method, reduction, value, *(buff[key] for buff in buffers))
            for key, value in buffer.items())
    elif isinstance(buffer, list):
        return reduction(buffer_multi_map_reduce(method, reduction, *values) for values in zip(buffer, *buffers))
    else:
        return method(buffer, *buffers)


def buffer_safe_dual_map(method, buffer, other):
    if isinstance(buffer, dict):
        if isinstance(other, dict):
            return {key: buffer_safe_dual_map(method, value, other[key]) for key, value in buffer.items()}
        else:
            return {key: buffer_safe_dual_map(method, value, other) for key, value in buffer.items()}
    elif isinstance(buffer, list):
        if isinstance(other, list):
            return [buffer_safe_dual_map(method, value, other_value) for value, other_value in zip(buffer, other)]
        else:
            return [buffer_safe_dual_map(method, value, other) for value in buffer]
    else:
        return method(buffer, other)


def default_buffer_type(buffer_shape: ShapeBufferType):
    if isinstance(buffer_shape, dict):
        return {key: default_buffer_type(shape) for key, shape in buffer_shape.items()}
    elif isinstance(buffer_shape, list):
        return [default_buffer_type(shape) for shape in buffer_shape]
    else:
        return torch.float32


def make_buffer_shape_type(buffer_shape: ShapeBufferType, buffer_type: DtypeBufferType):
    buffer_shape = (buffer_shape,) if isinstance(buffer_shape, int) else buffer_shape
    buffer_type = default_buffer_type(buffer_shape) if buffer_type is None else buffer_type

    return buffer_shape, buffer_type


def check_buffer(buffer: BufferType, buffer_shape: ShapeBufferType, buffer_type: DtypeBufferType) -> int:
    """ Checks that the buffer matches the definition and returns the batch size. """
    if isinstance(buffer, dict):
        if isinstance(buffer_shape, dict) and isinstance(buffer_type, dict):
            return max(check_buffer(buffer[key], buffer_shape[key], buffer_type[key]) for key in buffer)
        else:
            raise ValueError("Incompatible Buffer")

    elif isinstance(buffer, list):
        if isinstance(buffer_shape, list) and isinstance(buffer_type, list):
            return max(check_buffer(*param) for param in zip(buffer, buffer_shape, buffer_type))
        else:
            raise ValueError("Incompatible Buffer")

    size, shape = buffer.shape[0], buffer.shape[1:]

    if shape != buffer_shape:
        raise ValueError("Incompatible Buffer")

    if buffer.dtype != buffer_type:
        raise ValueError("Incompatible Buffer")

    return size


def buffer_size(buffer: BufferType) -> int:
    """ Unsafe function that checks the batch size of a buffer. Assumes identical batch sizes and sane structure. """
    if isinstance(buffer, dict):
        return max(buffer_size(buff) for buff in buffer.values())
    elif isinstance(buffer, list):
        return max(buffer_size(buff) for buff in buffer)
    else:
        return buffer.shape[0]


def make_buffer(size: int,
                shape: Union[Dict, List, Tuple],
                dtype: Union[Dict, List, torch.dtype],
                device: Union[str, torch.device] = 'shared') -> BufferType:
    """ Create a dynamically structured PyTorch buffer.

    The shape parameter may be a single shape, a list of shapes in order, or a dictionary of named shapes.
    The types parameter must have the same structure.

    Parameters
    ----------
    size: int
        Size of the first dimension for each tensor.
    shape: {tuple, list, dict}
        The shape of the other dimensions for each tensor.
    dtype: {tuple, list, dict}
        The type of each buffer, must have the same structure as buffer_shape
    device: str
        Which device to place the buffer on. Supported options are {'cpu', 'shared', 'pin', 'cuda:n'}
    """
    # Dictionary of shapes / types
    if isinstance(shape, dict):
        assert isinstance(dtype, dict)

        return {
            name: make_buffer(size, shape, dtype[name], device=device)
            for name, shape in shape.items()
        }

    # List of shapes / types
    if isinstance(shape, list):
        assert isinstance(dtype, list)

        return [make_buffer(size, shape, dtype, device=device)
                for shape, dtype in zip(shape, dtype)]

    # Single shape / type
    else:
        tensor = torch.empty((size, *shape), dtype=dtype)
        if device == 'shared':
            tensor.share_memory_()
            return tensor
        elif device == 'pin':
            return tensor.pin_memory()
        else:
            return tensor.to(device)


def load_buffer(to_buffer: BufferType, from_buffer: BufferType, size: int, start_index: int = 0):
    """ Copy data from one buffer into another with a given size and offset.

    Parameters
    ----------
    to_buffer : PyTorch Buffer
        The destination buffer.
    from_buffer : PyTorch Buffer
        The source buffer. Must have the same structure as the destination buffer.
    size: int
        How many elements from each tensor to transfer.
    start_index: int
        The offset in the destination buffer from which to start writing.
    """
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            load_buffer(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            load_buffer(to_tensor, from_tensor, size, start_index)

    else:
        # noinspection PyUnresolvedReferences
        to_buffer[start_index:start_index + size].copy_(from_buffer[:size])


def load_buffer_safe(to_buffer: BufferType, from_buffer: BufferType, size: int, start_index: int = 0):
    """ Copy data from one buffer into another with a given size and offset.

    Parameters
    ----------
    to_buffer : PyTorch Buffer
        The destination buffer.
    from_buffer : PyTorch Buffer
        The source buffer. Must have the same structure as the destination buffer.
    size: int
        How many elements from each tensor to transfer.
    start_index: int
        The offset in the destination buffer from which to start writing.
    """
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            load_buffer_safe(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            load_buffer_safe(to_tensor, from_tensor, size, start_index)

    else:
        # noinspection PyUnresolvedReferences
        to_buffer[start_index:start_index + size].copy_(torch.as_tensor(from_buffer[:size]))


def set_buffer(to_buffer: BufferType, from_buffer: BufferType, index):
    if isinstance(to_buffer, dict):
        if isinstance(from_buffer, dict):
            for key, to_tensor in to_buffer.items():
                set_buffer(to_tensor, from_buffer[key], index)
        else:
            for key, to_tensor in to_buffer.items():
                set_buffer(to_tensor, from_buffer, index)

    elif isinstance(to_buffer, list):
        if isinstance(from_buffer, list):
            for to_tensor, from_tensor in zip(to_buffer, from_buffer):
                set_buffer(to_tensor, from_tensor, index)
        else:
            for to_tensor in to_buffer:
                set_buffer(to_tensor, from_buffer, index)

    else:
        if isinstance(from_buffer, np.ndarray):
            from_buffer = torch.from_numpy(from_buffer)
        to_buffer[index] = from_buffer


def unload_buffer(to_buffer, from_buffer, size: int, start_index: int = 0):
    """ Copy data from one buffer into another with a given size and offset.

    This function is very similar to load_buffer, just start index affects the offset of the source buffer
    instead of the destination buffer.

    Parameters
    ----------
    to_buffer : PyTorch Buffer
        The destination buffer.
    from_buffer : PyTorch Buffer
        The source buffer. Must have the same structure as the destination buffer.
    size: int
        How many elements from each tensor to transfer.
    start_index: int
        The offset in the source buffer from which to start reading.
    """
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            unload_buffer(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            unload_buffer(to_tensor, from_tensor, size, start_index)

    else:
        to_buffer[:size].copy_(from_buffer[start_index:start_index + size])


def zero_buffer(buffer: BufferType):
    if isinstance(buffer, dict):
        for key, tensor in buffer.items():
            zero_buffer(tensor)

    elif isinstance(buffer, list):
        for tensor in buffer:
            zero_buffer(tensor)

    else:
        buffer[:] = 0


def slice_buffer(buffer: BufferType, begin: int = 0, end: int = -1):
    """ Recursively slice a PyTorch Buffer.

    Parameters
    ----------
    buffer: PyTorch Buffer
        Buffer to slice.
    begin: int
        Start index of the slice.
    end: int
        End index of the slice.

    Returns
    -------

    """
    if isinstance(buffer, dict):
        return {key: slice_buffer(val, begin, end) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [slice_buffer(val, begin, end) for val in buffer]
    else:
        return buffer[begin:end]


def index_buffer(buffer: BufferType, indices: Union[int, np.ndarray]):
    if isinstance(buffer, dict):
        return {key: index_buffer(val, indices) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [index_buffer(val, indices) for val in buffer]
    else:
        return buffer[indices]


def send_buffer(buffer: BufferType, device: str):
    """ Transfer a buffer to another device.

    Parameters
    ----------
    buffer: PyTorch Buffer
        The buffer to transfer.
    device: str
        Target device.
    """
    if isinstance(buffer, dict):
        return {key: send_buffer(val, device) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [send_buffer(val, device) for val in buffer]
    else:
        return buffer.to(device)


def raw_buffer(value) -> BufferType:
    if isinstance(value, Buffer):
        return value.buffer

    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)

    else:
        return value


def raw_buffer_and_size(value, size: int = None) -> Tuple[BufferType, int]:
    if isinstance(value, Buffer):
        return value.buffer, value.size

    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value), value.shape[0]

    elif size is None:
        return value, buffer_size(value)

    else:
        return value, size
