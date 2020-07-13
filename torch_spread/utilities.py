import pickle
import signal
import torch
import os

import numpy as np
from io import BytesIO
from multiprocessing.reduction import ForkingPickler

from lz4 import frame
import msgpack

from typing import Tuple, List, Union, Dict, Sized, Iterable, Generator, Optional, Any, Hashable

# Recursive definitions not supported yet, so I use Any for the subtypes
# But the correct type should have BufferType where Any is.
BufferType = Union[torch.Tensor, List[Any], Dict[Hashable, Any]]

DEBUG = False
VERBOSE = False


def serialize_int(x: int) -> bytes:
    """ Efficiently convert a python integer to a serializable bytestring. """
    byte_size = (int.bit_length(x) + 8) // 8
    return int.to_bytes(x, length=byte_size, byteorder='big')


def deserialize_int(b: bytes) -> int:
    """ Recover integer value from bytestring. """
    return int.from_bytes(b, byteorder='big')


def serialize_tensor(tensor) -> bytes:
    """ Serialize a tensor by placing it in shared memory. """
    buf = BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
    return buf.getvalue()


def deserialize_tensor(serialized_tensor: bytes):
    """ Convert back from a serialize tensor. Not that this can only be done once for a given bytestring. """
    return pickle.loads(serialized_tensor)


def make_buffer(buffer_size: int, buffer_shape: Union[Dict, List, Tuple], buffer_type: Union[Dict, List, torch.dtype],
                device: Union[str, torch.device] = 'shared') -> BufferType:
    """ Create a dynamically structured PyTorch buffer.

    The shape parameter may be a single shape, a list of shapes in order, or a dictionary of named shapes.
    The types parameter must have the same structure.

    Parameters
    ----------
    buffer_size: int
        Size of the first dimension for each tensor.
    buffer_shape: {tuple, list, dict}
        The shape of the other dimensions for each tensor.
    buffer_type: {tuple, list, dict}
        The type of each buffer, must have the same structure as buffer_shape
    device: str
        Which device to place the buffer on. Supported options are {'cpu', 'shared', 'pin', 'cuda:n'}
    """
    # Dictionary of shapes / types
    if isinstance(buffer_shape, dict):
        assert isinstance(buffer_type, dict)

        return {
            name: make_buffer(buffer_size, shape, buffer_type[name], device=device)
            for name, shape in buffer_shape.items()
        }

    # List of shapes / types
    if isinstance(buffer_shape, list):
        assert isinstance(buffer_shape, list)

        return [make_buffer(buffer_size, shape, type, device=device)
                for shape, type in zip(buffer_shape, buffer_type)]

    # Single shape / type
    else:
        tensor = torch.empty((buffer_size, *buffer_shape), dtype=buffer_type)
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
        to_buffer[start_index:start_index + size].copy_(from_buffer[:size])


def load_numpy_buffer(to_buffer: BufferType, from_buffer: BufferType, size: int, start_index: int = 0):
    """ Copy data from one buffer into another with a given size and offset.

    This function is very similar to load_buffer, but the from_buffer is a numpy buffer instead of a torch buffer.

    Parameters
    ----------
    to_buffer : PyTorch Buffer
        The destination buffer.
    from_buffer : Numpy Buffer
        The source buffer. Must have the same structure as the destination buffer.
    size: int
        How many elements from each tensor to transfer.
    start_index: int
        The offset in the destination buffer from which to start writing.
    """
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            load_numpy_buffer(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            load_numpy_buffer(to_tensor, from_tensor, size, start_index)

    else:
        to_buffer[start_index:start_index + size].copy_(torch.from_numpy(from_buffer)[:size])


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


def send_sigkill(pid: int):
    """ Forcefully kill a process by PID.

    Parameters
    ----------
    pid: int
        Process to kill
    """
    os.kill(pid, signal.SIGKILL)


def relative_channel(base_channel: str, prefix: str) -> str:
    """ Convert an IPC channel into one that is prefixed by a temporary directory.

    This allows us to make multiple NetworkManagers that dont interfere with each other.

    Parameters
    ----------
    base_channel: str
        The IPC channel to start with
    prefix : str
        The prefix for the IPC path.

    Returns
    -------
    str
        The updated relative path.

    """
    protocol, path = base_channel.split("//")
    return "{}//{}/{}".format(protocol, prefix, path)


def iterate_window(iterator: Union[Sized, Iterable], n: int = 2) -> Generator:
    """ Iterate over a sliding window of an iterator with no overlap.

    Parameters
    ----------
    iterator: Iterable
        The target to iterate over. Must support __len__. Total length must be a multiple of n.
    n: int
        Window size

    Returns
    -------
    Iterable
        Generator for viewing every n elements of the iterator.
    """
    size = len(iterator)
    iterator = iter(iterator)
    for _ in range(0, size, n):
        yield (next(iterator) for _ in range(n))


def optional(variable: Optional[object], default: object):
    """ Optional parameter that is default if None.
    """
    return default if variable is None else variable


def _serialize_compressed_buffer(buffer: BufferType, compress: int):
    if isinstance(buffer, dict):
        return {key: _serialize_compressed_buffer(val, compress) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [_serialize_compressed_buffer(val, compress) for val in buffer]
    elif isinstance(buffer, torch.Tensor):
        buffer = buffer.numpy()

    pickled = buffer.view(np.uint8).data
    pickled = frame.compress(pickled, compression_level=compress)

    shape = buffer.shape
    dtype = buffer.dtype.str

    return dtype, shape, pickled


def _serialize_uncompressed_buffer(buffer: BufferType):
    if isinstance(buffer, dict):
        return {key: _serialize_uncompressed_buffer(val) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [_serialize_uncompressed_buffer(val) for val in buffer]
    elif isinstance(buffer, torch.Tensor):
        buffer = buffer.numpy()

    pickled = buffer.view(np.uint8).data
    shape = buffer.shape
    dtype = buffer.dtype.str

    return dtype, shape, pickled


def serialize_buffer(buffer: BufferType, compress: int = -1):
    if compress < 0:
        serialized = _serialize_uncompressed_buffer(buffer)
    else:
        serialized = _serialize_compressed_buffer(buffer, compress)

    serialized = msgpack.dumps((compress, serialized), use_bin_type=True)
    return serialized


def _deserialize_compressed_buffer(serialized: bytes):
    if isinstance(serialized, dict):
        return {key: _deserialize_compressed_buffer(val) for key, val in serialized.items()}
    elif not isinstance(serialized[0], str):
        return [_deserialize_compressed_buffer(val) for val in serialized]

    dtype, shape, pickled = serialized
    array = np.frombuffer(frame.decompress(pickled), np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(array)


def _deserialize_uncompressed_buffer(serialized: bytes):
    if isinstance(serialized, dict):
        return {key: _deserialize_uncompressed_buffer(val) for key, val in serialized.items()}
    elif not isinstance(serialized[0], str):
        return [_deserialize_uncompressed_buffer(val) for val in serialized]

    dtype, shape, pickled = serialized
    array = np.frombuffer(pickled, np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(array)


def deserialize_buffer(serialized: bytes):
    compress, serialized = msgpack.loads(serialized, raw=False, use_list=False)
    if compress < 0:
        return _deserialize_uncompressed_buffer(serialized)
    else:
        return _deserialize_compressed_buffer(serialized)


def _deserialize_compressed_buffer_into(to_buffer: BufferType, serialized: bytes):
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            size = _deserialize_compressed_buffer_into(to_tensor, serialized[key])
        return size

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, serialized):
            size = _deserialize_compressed_buffer_into(to_tensor, from_tensor)
        return size

    else:
        dtype, shape, pickled = serialized
        from_buffer = np.frombuffer(frame.decompress(pickled), np.dtype(dtype)).reshape(shape)
        from_buffer = torch.from_numpy(from_buffer)

        size = shape[0]
        to_buffer[:size].copy_(from_buffer)

        return size


def _deserialize_uncompressed_buffer_into(to_buffer: BufferType, serialized: bytes):
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            size = _deserialize_uncompressed_buffer_into(to_tensor, serialized[key])
        return size

    elif isinstance(to_buffer, (list, tuple)):
        for to_tensor, from_tensor in zip(to_buffer, serialized):
            size = _deserialize_uncompressed_buffer_into(to_tensor, from_tensor)
        return size

    else:
        dtype, shape, pickled = serialized
        from_buffer = np.frombuffer(pickled, np.dtype(dtype)).reshape(shape)
        from_buffer = torch.from_numpy(from_buffer)

        size = shape[0]
        to_buffer[:size].copy_(from_buffer)

        return size


def deserialize_buffer_into(to_buffer: BufferType, serialized: bytes):
    compress, serialized = msgpack.loads(serialized, raw=False, use_list=False)
    if compress < 0:
        size = _deserialize_uncompressed_buffer_into(to_buffer, serialized)
    else:
        size = _deserialize_compressed_buffer_into(to_buffer, serialized)
    return size, compress