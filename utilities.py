import pickle
import torch
import numpy as np

from io import BytesIO
from multiprocessing.reduction import ForkingPickler

from multiprocessing import Array
from typing import Tuple, List

import signal
import os

DEBUG = False
VERBOSE = False


def torch_dtype_to_numpy(dtype):
    return getattr(np, str(dtype).split('.')[1])


def numpy_dtype_to_ctype(dtype):
    return np.ctypeslib.as_ctypes_type(dtype)


def torch_dtype_to_ctype(dtype):
    return numpy_dtype_to_ctype(numpy_dtype_to_ctype(dtype))


def serialize_int(x: int):
    byte_size = (int.bit_length(x) + 8) // 8
    return int.to_bytes(x, length=byte_size, byteorder='big')


def deserialize_int(b: bytes):
    return int.from_bytes(b, byteorder='big')


def serialize_tensor(tensor):
    buf = BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
    return buf.getvalue()


def deserialize_tensor(serialized_tensor: bytes):
    return pickle.loads(serialized_tensor)


def shared_array(ctype: type, shape: Tuple[int, ...], lock: bool = False):
    buffer_base = Array(ctype, int(np.prod(shape)), lock=lock)
    buffer = np.ctypeslib.as_array(buffer_base).reshape(shape)

    return buffer


def make_buffer(buffer_size, buffer_shape, buffer_type, device='shared'):
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
            return tensor.share_memory_()
        elif device == 'pin':
            return tensor.pin_memory()
        else:
            return tensor.to(device)


def load_buffer(to_buffer, from_buffer, size: int, start_index: int = 0):
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            load_buffer(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, list):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            load_buffer(to_tensor, from_tensor, size, start_index)

    else:
        to_buffer[start_index:start_index + size].copy_(from_buffer[:size])
        # to_buffer[start_index:start_index + size] = from_buffer[:size]


def unload_buffer(to_buffer, from_buffer, size: int, start_index: int = 0):
    if isinstance(to_buffer, dict):
        for key, to_tensor in to_buffer.items():
            unload_buffer(to_tensor, from_buffer[key], size, start_index)

    elif isinstance(to_buffer, list):
        for to_tensor, from_tensor in zip(to_buffer, from_buffer):
            unload_buffer(to_tensor, from_tensor, size, start_index)

    else:
        to_buffer[:size].copy_(from_buffer[start_index:start_index + size])
        # to_buffer[:size] = from_buffer[start_index:start_index + size]


def slice_buffer(buffer, begin=0, end=-1):
    if isinstance(buffer, dict):
        return {key: slice_buffer(val, begin, end) for key, val in buffer.items()}
    elif isinstance(buffer, list):
        return [slice_buffer(val, begin, end) for val in buffer]
    else:
        return buffer[:end]


def send_buffer(buffer, device):
    if isinstance(buffer, dict):
        return {key: send_buffer(val, device) for key, val in buffer.items()}
    elif isinstance(buffer, list):
        return [send_buffer(val, device) for val in buffer]
    else:
        return buffer.to(device)

def send_sigkill(pid):
    os.kill(pid, signal.SIGKILL)


def relative_channel(base_channel, identity):
    protocol, path = base_channel.split("//")
    return "{}//{}/{}".format(protocol, identity, path)


def iterate_window(iterator: List, n: int = 2):
    size = len(iterator)
    iterator = iter(iterator)
    for _ in range(0, size, n):
        yield (next(iterator) for _ in range(n))


def optional(variable, default):
    return default if variable is None else variable