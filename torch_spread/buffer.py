from typing import Optional, Union, Callable, Hashable

import torch
from torch import Tensor

from .utilities import ShapeBufferType, DtypeBufferType, BufferType
from .buffer_tools import buffer_map, buffer_safe_dual_map, buffer_multi_map_reduce
from .buffer_tools import make_buffer, load_buffer, make_buffer_shape_type, buffer_fill_information
from .buffer_tools import zero_buffer, send_buffer, index_buffer, set_buffer, raw_buffer, iterate_buffer


# noinspection SpellCheckingInspection
def wrap_operators(cls):
    """ Dynamically add all of the arithmetic methods from torch tensors to the buffer class. """

    # Two-argument arithmetic operators
    # ---------------------------------------------------------------------------------------------
    for mm in ['__add__', '__sub__', '__mul__',
               '__floordiv__', '__div__', '__mod__', '__divmod__', '__pow__',
               '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',
               '__radd__', '__rsub__', '__rmul__',
               '__rfloordiv__', '__rdiv__', '__rmod__', '__rdivmod__', '__rpow__',
               '__rlshift__', '__rrshift__', '__rand__', '__ror__', '__rxor__',
               '__iadd__', '__isub__', '__imul__',
               '__ifloordiv__', '__idiv__', '__imod__', '__idivmod__', '__ipow__',
               '__ilshift__', '__irshift__', '__iand__', '__ior__', '__ixor__']:
        def make_operator(operator):

            # noinspection PyShadowingNames
            def operator_method(self, other):
                if isinstance(other, cls):
                    other = other.buffer

                torch_operator = getattr(torch.Tensor, operator)
                return cls.from_buffer(buffer_safe_dual_map(torch_operator, self.buffer, other),
                                       self.shape, self.dtype, self.size)

            return operator_method

        operator_method = make_operator(mm)
        operator_method.__name__ = mm
        setattr(cls, mm, operator_method)

    # Unitary arithmetic operators
    # ---------------------------------------------------------------------------------------------
    for mm in ['__pos__', '__neg__', '__abs__', '__invert__', '__round__',
               '__floor__', '__ceil__', '__trunc__']:
        def make_operator(operator):

            # noinspection PyShadowingNames
            def operator_method(self):
                torch_operator = getattr(torch.Tensor, operator)
                return cls.from_buffer(buffer_map(torch_operator, self.buffer),
                                       self.shape, self.dtype, self.size)

            return operator_method

        operator_method = make_operator(mm)
        operator_method.__name__ = mm
        setattr(cls, mm, operator_method)

    return cls


@wrap_operators
class Buffer:
    def __init__(self,
                 shape: ShapeBufferType,
                 dtype: DtypeBufferType,
                 size: int,
                 device: Optional[str] = 'cpu',
                 zero: bool = True):
        """ Helper object for managing buffer objects in a convenient manner.

        This is essentially a fancy wrapper around recursive dictionaries and lists of tensors.
        This class implements all of the PyTorch tensor mathematical operators,
        so you should be able to use it as a tensor for most operations.

        Notes
        -----
        Note that a buffer has an explicit separation between the batch size dimension and the shape of the base buffer.
        This is because the batch dimension must be the same between all elements in the buffer in order
        to efficiently store and predict on it.

        Note that, when indexing, the buffer may return a single element. In that case, the size will be set to 0.

        Parameters
        ----------
        shape: ShapeBufferType
            Defines the base shape for the buffer. Note that this does not include the batch size dimension.
        dtype: DtypeBufferType
            Defines the torch data type for each element in the buffer.
        size: int
            The size of the 'batch size' dimension in the buffer.
        device: str
            The location to place the buffer at. This can be 'cpu', 'cuda:#', or 'shared'
            If `None`, then the buffer will not be created by the class and must be assigned separately.
            This allows for copy-less assignment (See Buffer.from_buffer).
        zero: bool
            Whether or not to zero out the buffer after creation. If False, it will have arbitrary data.
        """
        self.size = size
        self.device = device
        self.shape, self.dtype = make_buffer_shape_type(shape, dtype)

        if device is not None:
            self.buffer = make_buffer(size, self.shape, self.dtype, device)

            if zero:
                zero_buffer(self.buffer)

    @staticmethod
    def from_buffer(buffer: Union[BufferType, "Buffer"],
                    shape: Optional[ShapeBufferType] = None,
                    dtype: Optional[DtypeBufferType] = None,
                    size: Optional[int] = None) -> "Buffer":
        """ Create a Buffer object from an existing buffer structure. (Tensor, list of Tensors, dict of Tensors, ...)

        This will avoid copying data, and will most likely return a simple wrapped data structure.

        Notes
        -----
        In order to ensure maximum performance, the optional shape and datatype parameters are not verified!
        If you provide incorrect buffer information, then the buffer object will become corrupt.

        Parameters
        ----------
        buffer: BufferType
            The raw buffer data to convert into a Buffer object.

        shape: ShapeBufferType, optional
        dtype: DtypeBufferType, optional
        size: int, optional
            Optional additional information about the buffer if they are known.
            These will slightly speed up creating if they are known.
            Otherwise the parameters will be determined from the buffer data.

        Returns
        -------
        Buffer:
            The wrapped buffer object.
        """

        if isinstance(buffer, Buffer):
            return buffer

        shape, dtype, size = buffer_fill_information(buffer, shape, dtype, size)

        new_buffer = Buffer(shape, dtype, size, None)
        new_buffer.buffer = buffer

        return new_buffer

    @property
    def base(self) -> BufferType:
        """ Access the raw buffer data of the buffer object.

        Returns
        -------
        BufferType
        """
        return self.buffer

    def clone(self, device: Optional[str] = None) -> "Buffer":
        """ Clone the buffer into a new memory location. Optionally move the object to a different device.

        Parameters
        ----------
        device: str, optional
            The Torch device to send the tensor to.

        Returns
        -------
        Buffer
        """
        if device is None:
            device = self.device

        new_buffer = Buffer(self.shape, self.dtype, self.size, device, zero=False)
        load_buffer(new_buffer.base, self.base, self.size)

        return new_buffer

    def to(self, device: str, inplace: bool = True) -> "Buffer":
        """ Send buffer data to a different device. Avoids copying if the device is the same as the current one.

        Parameters
        ----------
        device: str
            Device string. See constructor for details.
        inplace: bool
            If True, then the buffer will be moved to the device inside of this object. The old data will be deleted.
            If False, then the buffer will create a new buffer and return it.

        Returns
        -------
        Buffer
        """
        if device == self.device:
            return self

        elif inplace:
            self.buffer = send_buffer(self.buffer, device)
            return self

        else:
            return self.clone(device)

    def map(self,
            torch_function: Callable[[Tensor], Tensor],
            output_shape: Optional[ShapeBufferType] = None,
            output_type: Optional[DtypeBufferType] = None,
            output_size: Optional[int] = None) -> "Buffer":
        """ A generalization of the python `map` function that gets applied uniformly to all elements of the buffer.

        Notes
        -----
        In order to ensure maximum performance, the optional shape and datatype parameters are not verified!
        If you provide incorrect buffer information, then the buffer object will become corrupt.

        Parameters
        ----------
        torch_function: (Tensor, ) -> Tensor
            The function that you want to apply to all of the internal tensors.

        output_shape: ShapeBufferType, optional
        output_type: DtypeBufferType, optional
        output_size: int, optional
            Optional additional information about the buffer if they are known.
            These will slightly speed up creation if they are known.
            Otherwise the parameters will be determined from the output buffer.

        Returns
        -------
        Buffer
        """
        return self.from_buffer(buffer_map(torch_function, self.buffer), output_shape, output_type, output_size)

    def map_preserve(self, torch_function: Callable[[Tensor], Tensor]) -> "Buffer":
        """ A shortcut method to `map` if you know that the function you are applying does not change the shape.

        Parameters
        ----------
        torch_function: (Tensor, ) -> Tensor
            The function that you want to apply to all of the internal tensors.
            You must ensure that the method does not modify the tensor shape or type.

        Returns
        -------
        Buffer
        """
        return self.map(torch_function, self.shape, self.dtype, self.size)

    def __getitem__(self, item) -> "Buffer":
        """ Index the `batch dimension` of the buffer tensors.

        Notes
        -----
        This function does not create a copy of the data!
        Any modifications to the indexed buffer will be reflected in the original buffer.

        Parameters
        ----------
        item: Index
            The index to be passed to all of the tensors in the buffer.
            This can be any type that tensor.__getitem__ accepts.

        Returns
        -------
        Buffer
        """
        new_size = None
        if isinstance(item, int):
            new_size = 0

        return self.from_buffer(index_buffer(self.buffer, item), self.shape, self.dtype, new_size)

    def __setitem__(self, key, value):
        """ Modify the buffer data while indexing by the `batch dimension` """
        set_buffer(self.buffer, raw_buffer(value), key)

    def __call__(self, *index, raw: bool = False) -> Union[BufferType, "Buffer"]:
        """ Index sub-tensors in the buffer.

        Parameters
        ----------
        index: 0 or more indices
            If the base buffer is a list, then these should be integer indices
            If the base buffer is a dictionary, then these should be the keys of the dictionary.
        raw: bool, optional
            If True, then this will return the base buffer data-structure.
            If False, then it will try and wrap the output in a Buffer object

        Returns
        -------
        BufferType:
            If the output is a Tensor or `raw` is True
        Buffer:
            If `raw` is False
        """
        if len(index) == 0:
            if isinstance(self.buffer, torch.Tensor) or raw:
                return self.buffer
            else:
                return self

        elif len(index) == 1:
            index = index[0]
            buffer = self.buffer[index]

            if isinstance(buffer, torch.Tensor) or raw:
                return buffer

            else:
                return self.from_buffer(buffer, self.shape[index], self.dtype[index], self.size)

        elif isinstance(self.buffer, dict):
            buffer = {idx: self.buffer[idx] for idx in index}
            buffer_shape = {idx: self.shape[idx] for idx in index}
            buffer_type = {idx: self.dtype[idx] for idx in index}
            return self.from_buffer(buffer, buffer_shape, buffer_type, self.size)

        elif isinstance(self.buffer, list):
            buffer = [self.buffer[idx] for idx in index]
            buffer_shape = [self.shape[idx] for idx in index]
            buffer_type = [self.dtype[idx] for idx in index]
            return self.from_buffer(buffer, buffer_shape, buffer_type, self.size)

        else:
            return self.buffer[index]

    def insert(self,
               index: Union[int, Hashable],
               buffer: Optional[Union[BufferType, "Buffer"]] = None,
               shape: Optional[ShapeBufferType] = None,
               dtype: Optional[DtypeBufferType] = None,
               size: Optional[int] = None):
        """ Insert a new element into the buffer.

        Notes
        -----
        In order to ensure maximum performance, the optional shape and datatype parameters are not verified!
        If you provide incorrect buffer information, then the buffer object will become corrupt.

        Parameters
        ----------
        index: int or dictionary key
            If the base is a list, then this should be the index of the new element.
            If the base is a dict, then this should be the key of the new element.
        buffer: BufferType or Buffer, optional
            The data of the new buffer to insert.
            If None, then you must provide shape and dtype information,
            and a new arbitrary buffer will be created.

        shape: ShapeBufferType, optional
        dtype: DtypeBufferType, optional
        size: int, optional
            Buffer information for the incoming data.
            If `buffer` is a Buffer object, then these will be determined from the object.
            If None, then these will be computed from the data structure.
            If provided, then the buffer will use the provided values.
            Size must match the size of the current buffer.
        """

        # Extract Buffer object information
        if isinstance(buffer, Buffer):
            shape = buffer.shape
            dtype = buffer.dtype
            size = buffer.size
            buffer = buffer.base

        # Determine the size of the new buffer
        if size is None:
            size = self.size

        elif size != self.size:
            raise ValueError("New buffer does not match the size of the existing buffers.")

        # Determine the new buffer shape and type information
        # Or create the buffer if no data has been provided
        if buffer is None:
            if None in (shape, dtype):
                raise ValueError("If buffer is None when inserting, must provide shape and type information.")

            buffer = make_buffer(size, shape, dtype, self.device)

        else:
            shape, dtype, size = buffer_fill_information(buffer, shape, dtype, size)

        # Insert the new buffer information
        if isinstance(self.buffer, dict):
            for storage, data in ((self.buffer, buffer), (self.dtype, dtype), (self.shape, shape)):
                storage[index] = data

        elif isinstance(self.buffer, list):
            for storage, data in ((self.buffer, buffer), (self.dtype, dtype), (self.shape, shape)):
                storage.insert(index, data)

        else:
            raise ValueError("Cannot insert a new Tensor into a single Tensor buffer.")

    def __repr__(self):
        return "Buffer:\n" + self.buffer.__repr__()

    def __str__(self):
        return "Buffer:\n" + self.buffer.__str__()

    def __len__(self):
        return self.size

    def __eq__(self, other: "Buffer"):
        def tensor_equality(x: Tensor, y: Tensor):
            return torch.all(torch.eq(x, y)).item()

        return buffer_multi_map_reduce(tensor_equality, all, self.buffer, other.buffer)

    def __iter__(self):
        return iterate_buffer(self.buffer)
