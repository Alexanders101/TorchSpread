import os
import pickle
import signal
from io import BytesIO
from typing import Tuple, List, Union, Dict, Sized, Iterable, Generator, Optional, Hashable

import msgpack
import numpy as np
import torch
from lz4 import frame
from multiprocessing.reduction import ForkingPickler

# Recursive definitions not supported yet, so I use Tensor for the subtypes
# But the correct type should have BufferType where Tensor is.
BufferType = Union[torch.Tensor, List[torch.Tensor], Dict[Hashable, torch.Tensor]]
ShapeBufferType = Union[Tuple[int, ...], List[Tuple[int, ...]], Dict[Hashable, Tuple[int, ...]]]
DtypeBufferType = Union[torch.dtype, List[torch.dtype], Dict[Hashable, torch.dtype]]

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

    shape = buffer.shape
    dtype = buffer.dtype.str

    if len(shape) > 0:
        pickled = buffer.view(np.uint8).data
    else:
        pickled = buffer.tobytes()
        
    pickled = frame.compress(pickled, compression_level=compress)

    return dtype, shape, pickled


def _serialize_uncompressed_buffer(buffer: BufferType):
    if isinstance(buffer, dict):
        return {key: _serialize_uncompressed_buffer(val) for key, val in buffer.items()}
    elif isinstance(buffer, (list, tuple)):
        return [_serialize_uncompressed_buffer(val) for val in buffer]
    elif isinstance(buffer, torch.Tensor):
        buffer = buffer.numpy()

    shape = buffer.shape
    dtype = buffer.dtype.str

    if len(shape) > 0:
        pickled = buffer.view(np.uint8).data
    else:
        pickled = buffer.tobytes()

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
