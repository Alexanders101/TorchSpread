import ctypes

import numpy as np

from .BufferTools import make_buffer_shape_type, check_buffer, load_buffer, load_buffer_safe, Buffer, raw_buffer, \
    raw_buffer_and_size
from .utilities import BufferType, ShapeBufferType, DtypeBufferType, mp_ctx


class BufferFixedQueue:
    def __init__(self,
                 buffer_shape: ShapeBufferType,
                 buffer_type: DtypeBufferType,
                 maxsize: int,
                 zero: bool = True):
        """ A fixed-size shared memory buffer queue. This is useful for workers to dump their states to.

        Parameters
        ----------
        buffer_shape: ShapeBufferType
            The shape structure for the stored buffer.
        buffer_type: DtypeBufferType
            The dtype structure for the stored buffer.
        maxsize: int
            Maximum size for this buffer. This size will be allocated at the start.
            Make sure the size can fit in shared memory!
        zero: bool
            If true, then the buffer will be set to 0 originally. If false than it will have arbitrary data.
        """
        self.maxsize = maxsize
        self.current_index = mp_ctx.Value(ctypes.c_int64, lock=False)

        self.buffer_shape, self.buffer_type = make_buffer_shape_type(buffer_shape, buffer_type)
        self.buffer = Buffer(self.buffer_shape, self.buffer_type, maxsize, device='shared', zero=zero)

        self.lock = mp_ctx.Lock()

    @property
    def size(self):
        with self.lock:
            return self.current_index.value

    def reset(self):
        with self.lock:
            self.current_index.value = 0

    def _get_index(self, size):
        with self.lock:
            current_index = self.current_index.value
            new_size = current_index + size
            if new_size > self.maxsize:
                return -1, -1

            self.current_index.value = new_size
            return current_index, new_size

    def put(self, buffer: BufferType, size: int = None):
        buffer, size = raw_buffer_and_size(buffer, size)

        # Lock the queue and get our write index. This function is synchronized.
        start_index, end_index = self._get_index(size)

        if start_index < 0:
            raise ValueError("Queue full.")

        # Once we have the safely selected indices, then the data transfer no longer needs to be locked!
        load_buffer_safe(self.buffer.base, buffer, size, start_index)

    def put_safe(self, buffer: BufferType):
        buffer = raw_buffer(buffer)
        size = check_buffer(buffer, self.buffer_shape, self.buffer_type)

        self.put(buffer, size)

    def get(self, start_index: int = None, end_index: int = None):
        if start_index is None:
            return self.buffer[:self.size]

        if end_index is None:
            end_index = start_index + 1

        return self.buffer[start_index:end_index]

    def __getitem__(self, item):
        return self.buffer[item]


class BufferRing:
    def __init__(self, buffer_shape: ShapeBufferType, buffer_type: DtypeBufferType, maxsize: int, zero: bool = True):
        """ An infinitely repeating ring buffer with a fixed maximum size.

        Parameters
        ----------
        buffer_shape: ShapeBufferType
            The shape structure for the stored buffer.
        buffer_type: DtypeBufferType
            The dtype structure for the stored buffer.
        maxsize: int
            Maximum size for this buffer. This size will be allocated at the start.
            Make sure the size can fit in shared memory!
        zero: bool
            If true, then the buffer will be set to 0 originally. If false than it will have arbitrary data.
        """
        self.maxsize = maxsize
        self.current_size = mp_ctx.Value(ctypes.c_int64, lock=False)
        self.current_index = mp_ctx.Value(ctypes.c_int64, lock=False)

        self.buffer_shape, self.buffer_type = make_buffer_shape_type(buffer_shape, buffer_type)
        self.buffer = Buffer(self.buffer_shape, self.buffer_type, maxsize, device='shared', zero=zero)

        self.indices = np.arange(maxsize)

        self.lock = mp_ctx.Lock()

    def reset(self):
        with self.lock:
            self.current_index.value = 0
            self.current_size.value = 0

    @property
    def size(self):
        with self.lock:
            return self.current_size.value

    def _get_index(self, size):
        with self.lock:
            current_index = self.current_index.value
            current_size = self.current_size.value

            new_index_raw = (current_index + size)
            new_index = new_index_raw % self.maxsize
            new_size = min(current_size + size, self.maxsize)

            self.current_index.value = new_index
            self.current_size.value = new_size

        return np.arange(current_index, new_index_raw) % self.maxsize

    def put(self, buffer: BufferType, size: int = None):
        buffer, size = raw_buffer_and_size(buffer, size)

        # Lock the queue and get our write index. This function is synchronized.
        indices = self._get_index(size)

        self.buffer[indices] = buffer

    def __getitem__(self, index):
        get_all = (isinstance(index, slice) and
                   index.start is None and
                   index.stop is None and
                   index.step is None)

        with self.lock:
            current_size = self.current_size.value
            start_index = self.current_index.value - current_size

        if get_all:
            index = slice(None, current_size)

        indices = (self.indices[index] + start_index) % self.maxsize

        if indices.max(initial=0) > current_size:
            raise IndexError()

        return self.buffer[indices]


class BufferFIFOQueue:
    def __init__(self,
                 buffer_shape: ShapeBufferType,
                 buffer_type: DtypeBufferType,
                 maxsize: int,
                 zero: bool = True):
        """ A fixed size, shared memory, First-In First-Out queue.
        
        Parameters
        ----------
        buffer_shape: ShapeBufferType
            The shape structure for the stored buffer.
        buffer_type: DtypeBufferType
            The dtype structure for the stored buffer.
        maxsize: int
            Maximum size for this buffer. This size will be allocated at the start.
            Make sure the size can fit in shared memory!
        """
        self.buffer_shape, self.buffer_type = make_buffer_shape_type(buffer_shape, buffer_type)
        self.buffer = Buffer(self.buffer_shape, self.buffer_type, maxsize, device='shared', zero=zero)

        self.maxsize = maxsize

        self._queue_size = mp_ctx.Value(ctypes.c_int64, lock=False)
        self._write_index = mp_ctx.Value(ctypes.c_int64, lock=False)
        self._read_index = mp_ctx.Value(ctypes.c_int64, lock=False)

        self.mutex = mp_ctx.Lock()
        self.not_empty = mp_ctx.Condition(self.mutex)
        self.not_full = mp_ctx.Condition(self.mutex)

    @property
    def size(self):
        with self.mutex:
            return self._queue_size.value

    def put(self, buffer, size: int = None):
        buffer, size = raw_buffer_and_size(buffer, size)

        with self.not_full:
            queue_size = self._queue_size.value
            while queue_size >= self.maxsize:
                self.not_full.wait()
                queue_size = self._queue_size.value

            write_index = self._write_index.value
            load_buffer(self.buffer.base, buffer, size, write_index)

            self._queue_size.value = queue_size + 1
            self._write_index.value = (write_index + 1) % self.maxsize

            self.not_empty.notify()

    def get(self, batch_size: int = 1):
        with self.not_empty:
            size = self._queue_size.value
            while size < batch_size:
                self.not_empty.wait()
                size = self._queue_size.value

            read_index = self._read_index.value
            read_indices = np.arange(read_index, read_index + batch_size) % self.maxsize
            item = self.buffer[read_indices].copy()

            self._queue_size.value = size - batch_size
            self._read_index.value = (read_index + batch_size) % self.maxsize

            self.not_full.notify()
            return item
