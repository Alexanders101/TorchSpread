import zmq
import pickle
import torch

from typing import Dict, Union, Optional, Any

from .NetworkManager import FrontendManager
from .NetworkSynchronization import SyncCommands, SynchronizationManager, relative_channel
from .utilities import make_buffer, serialize_tensor, serialize_int, slice_buffer, BufferType, serialize_buffer, deserialize_buffer
from abc import ABC, abstractmethod


class ClientBase(ABC):
    def __init__(self, config: Dict[str, Any], batch_size: int):
        self.config = config
        self.batch_size = batch_size

        self.identity = None
        self.predict_size = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deregister()

    def __call__(self, data):
        return self.predict(data)

    @property
    def connected(self) -> bool:
        """ Is this client as registered with a manager? """
        return self.identity is not None

    @property
    def predicting(self) -> bool:
        """ Does this client have an outstanding asynchronous prediction request? """
        return self.predict_size is not None

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def register(self) -> None:
        pass

    @abstractmethod
    def deregister(self) -> None:
        pass

    @abstractmethod
    def predict(self, data: Union[int, BufferType]) -> BufferType:
        pass

    @abstractmethod
    def predict_async(self, data: Union[int, BufferType]) -> None:
        pass

    @abstractmethod
    def receive_async(self) -> BufferType:
        pass


class NetworkClient(ClientBase):
    def __init__(self, config: Dict[str, Any], batch_size: int):
        """ Primary management class for any remote clients that use the network manager.

        Notes
        -----
        This class sets up the link between any client and the network manager. It sets up buffers for prediction and
        results and it sends prediction requests to the main cluster. This class is only intended to be used in
        a context management 'with' block. Manual register and deregister methods are provided but discouraged.

        Parameters
        ----------
        config: Dict
            Client configuration dictionary provided by the manager. Typically NetworkManager.client_config.

            This configuration is designed to be minimal and only composed on python data-types, meaning that it can
            be easily pickled and sent to threads, processes, and remote servers.

        batch_size: int
            Requested batch size for this client. This must at most the network batch size specified in the
            manager configuration. If you wish to use larger batch sizes, you must either increase the network
            batch size or manually perform multiple prediction calls.

        See Also
        --------
        torch_spread.NetworkManager.NetworkManager.client_config
        """
        super(NetworkClient, self).__init__(config, batch_size)

        assert batch_size <= config['batch_size'], "Client batch size is greater than manager batch size."

        self.ipc_dir = config['ipc_dir']
        self.device = 'shared'

        # Create this client's buffers
        self.input_buffer = make_buffer(batch_size, config["input_shape"], config["input_type"], self.device)
        self.output_buffer = make_buffer(batch_size, config["output_shape"], config["output_type"], self.device)

        # Create communication objects
        self.context = zmq.Context()

        self.synchronization_queue = self.context.socket(zmq.DEALER)
        self.synchronization_queue.connect(relative_channel(SynchronizationManager.SYNC_FRONTEND_CHANNEL, self.ipc_dir))
        self.request_queue = self.context.socket(zmq.DEALER)

    def __del__(self):
        if self.identity is not None:
            self.synchronization_queue.send_multipart([SyncCommands.DEREGISTER, self.identity])

    def register(self) -> None:
        """ Initialize the prediction session for this client. This is likely not supposed to be used explicitly. """
        if self.connected:
            raise AssertionError("Cannot register a client twice.")

        num_networks = self.config["num_networks"]
        buffers = [serialize_tensor([self.input_buffer, self.output_buffer]) for _ in range(num_networks)]
        self.synchronization_queue.send_multipart([SyncCommands.REGISTER, pickle.dumps(buffers)])
        for _ in range(self.config["num_networks"]):
            network, self.identity = self.synchronization_queue.recv_multipart()

        self.request_queue.setsockopt(zmq.IDENTITY, self.identity)
        self.request_queue.connect(relative_channel(FrontendManager.FRONTEND_CHANNEL, self.ipc_dir))

    def deregister(self) -> None:
        if not self.connected:
            raise AssertionError("Cannot deregister a client that has not been registered")

        self.synchronization_queue.send_multipart([SyncCommands.DEREGISTER, self.identity])
        for _ in range(self.config["num_networks"]):
            self.synchronization_queue.recv_multipart()

        self.identity = None

    def _load_input_buffer(self, data: BufferType, input_buffer: Optional[BufferType] = None) -> int:
        """ Copy over any buffer into the client's shared buffer with the manager.

        Parameters
        ----------
        data: BufferType
            The desired buffer to copy over
        input_buffer: Optional[BufferType]
            Used for recursive calls when copying dictionary or list buffers.

        Returns
        -------
        int
            Batch size of the created buffer
        """
        size = 0
        if input_buffer is None:
            input_buffer = self.input_buffer

        if isinstance(data, dict):
            for key, tensor in data.items():
                size = self._load_input_buffer(tensor, input_buffer[key])
            return size

        elif isinstance(data, (list, tuple)):
            for tensor, buffer in zip(data, input_buffer):
                size = self._load_input_buffer(tensor, buffer)
            return size

        else:
            if not torch.is_tensor(data):
                data = torch.from_numpy(data)

            size = data.size(0)
            input_buffer[:size].copy_(data)
            return size

    def predict_inplace_async(self, size: Optional[int] = None) -> None:
        assert self.connected, "Worker has tried to predict without registering first."
        assert not self.predicting, "Cannot launch two asynchronous prediction requests at once. " \
                                    "Must finish one before sending a new one."

        if size is None:
            size = self.batch_size

        self.predict_size = size
        self.request_queue.send(serialize_int(size))

    def predict_inplace(self, size: int = None):
        assert self.connected, "Worker has tried to predict without registering first."

        size = self.batch_size if size is None else size
        self.request_queue.send(serialize_int(size))

        self.request_queue.recv()
        return slice_buffer(self.output_buffer, 0, size)

    def predict(self, data: Union[int, BufferType]) -> BufferType:
        """ General prediction function for the client. Determines the correct type of prediction to make

        Parameters
        ----------
        data: int or BufferType
            If data is an integer, then we predict on the current input buffer in place with the given size.
            If data is a Tensor buffer, then we copy the tensor and predict on it
            If data is a numpy array buffer, we convert the array into a tensor and predict on it.

        Returns
        -------
        BufferType
            A view into the output buffer as a tensor buffer.

        Raises
        ------
        AssertionError
            If there has already been an asynchronous prediction request queued.
        """
        assert not self.predicting, "Cannot request a synchronous prediction if an async prediction has been queued"

        if isinstance(data, int):
            size = data
        else:
            size = self._load_input_buffer(data)

        return self.predict_inplace(size)

    def predict_async(self, data: Union[int, BufferType]) -> None:
        """ Launch an asynchronous prediction request. Sends data to the networks and returns.

        Parameters
        ----------
        data: A buffer of Numpy arrays, torch tensors, or integer size
            See NetworkClient.predict

        Raises
        ------
        AssertionError
            If there has already been an asynchronous prediction request queued.
        """
        assert not self.predicting, "Cannot launch two asynchronous prediction requests at once. " \
                                    "Must finish one before sending a new one."

        if isinstance(data, int):
            size = data
        else:
            size = self._load_input_buffer(data)

        self.predict_inplace_async(size)

    def receive_async(self) -> BufferType:
        """ Wait for the results of of an async prediction call to

        Returns
        -------
        BufferType
            A view into the output buffer as a tensor buffer.

        Raises
        ------
        AssertionError
            If there is no outstanding asynchronous prediction request.
        """
        assert self.predicting, "Cannot receive a result until launching an asynchronous prediction request."

        self.request_queue.recv()

        predict_size = self.predict_size
        self.predict_size = None
        return slice_buffer(self.output_buffer, 0, predict_size)


# This class has to be here to avoid circular imports
class RemoteCommands:
    REGISTER = b'R'
    DEREGISTER = b'D'
    ERROR = b'E'
    PREDICT = b'P'

    KILL = b'K'


class RemoteClient(ClientBase):
    def __init__(self, config: Dict[str, Any], batch_size: int, hostname: str, port: int):
        super(RemoteClient, self).__init__(config, batch_size)

        self.hostname = hostname
        self.port = port

        self.context = zmq.Context()
        self.request_queue = self.context.socket(zmq.REQ)

    def __del__(self):
        if self.connected:
            self.request_queue.send_multipart([RemoteCommands.DEREGISTER, self.identity])

    def register(self) -> None:
        if self.connected:
            raise AssertionError("Cannot register a client twice.")

        self.request_queue.connect(f"tcp://{self.hostname}:{self.port}")
        self.request_queue.send_multipart([RemoteCommands.REGISTER, serialize_int(self.batch_size)])
        result, self.identity = self.request_queue.recv_multipart()

        if result == RemoteCommands.ERROR:
            self.identity = None
            raise ConnectionError(f"Failed to register: {self.identity}")

    def deregister(self) -> None:
        if not self.connected:
            raise AssertionError("Cannot deregister a client that has not been registered")

        self.request_queue.send_multipart([RemoteCommands.DEREGISTER, self.identity])
        result, data = self.request_queue.recv_multipart()

        if result == RemoteCommands.ERROR:
            raise ConnectionError(f"Failed to deregister: {data}")

        self.identity = None

    def predict(self, data: Union[int, BufferType]) -> BufferType:
        if isinstance(data, int):
            raise ValueError("RemoteClient does not support in-place prediction.")

        assert self.connected, "Worker has tried to predict without registering first."
        assert not self.predicting, "Cannot request a synchronous prediction if an async prediction has been queued"

        self.request_queue.send_multipart([RemoteCommands.PREDICT, serialize_buffer(data)])
        response, data = self.request_queue.recv_multipart()

        if response == RemoteCommands.ERROR:
            raise ValueError(data)
        else:
            return deserialize_buffer(data)

    def predict_async(self, data: Union[int, BufferType]) -> None:
        if isinstance(data, int):
            raise ValueError("RemoteClient does not support in-place prediction.")

        assert self.connected, "Worker has tried to predict without registering first."
        assert not self.predicting, "Cannot launch two asynchronous prediction requests at once. " \
                                    "Must finish one before sending a new one."

        self.predict_size = 1
        self.request_queue.send_multipart([RemoteCommands.PREDICT, serialize_buffer(data)])

    def receive_async(self) -> BufferType:
        assert self.connected, "Worker has tried to predict without registering first."
        assert self.predicting, "Cannot receive a result until launching an asynchronous prediction request."

        response, data = self.request_queue.recv_multipart()

        self.predict_size = None

        if response == RemoteCommands.ERROR:
            raise ValueError(data)
        else:
            return deserialize_buffer(data)