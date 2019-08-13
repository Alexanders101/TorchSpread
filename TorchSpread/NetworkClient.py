import zmq
import pickle
import torch
import numpy as np

from typing import Dict, Union, Optional
from .utilities import make_buffer, serialize_tensor, serialize_int, load_buffer, slice_buffer

from .NetworkManager import ResponseManager, RequestManager
from .NetworkSynchronization import SyncCommands, SynchronizationManager, relative_channel


class NetworkClient:
    def __init__(self, config: Dict, batch_size: int):
        assert batch_size <= config['batch_size'], "Client batch size is greater than manager batch size."

        self.config = config
        self.ipc_dir = config['ipc_dir']
        self.device = 'shared'

        # Create this client's buffers
        self.batch_size = batch_size
        self.input_buffer = make_buffer(batch_size, config["input_shape"], config["input_type"], self.device)
        self.output_buffer = make_buffer(batch_size, config["output_shape"], config["output_type"], self.device)

        # Create communication objects
        self.context = zmq.Context()

        self.request_queue = self.context.socket(zmq.PUSH)
        self.response_queue = self.context.socket(zmq.DEALER)

        self.synchronization_queue = self.context.socket(zmq.DEALER)
        self.synchronization_queue.connect(relative_channel(SynchronizationManager.SYNC_FRONTEND_CHANNEL, self.ipc_dir))

        self.identity = None
        self.predict_size = None

    def register(self):
        if self.connected:
            raise ValueError("Cannot register a client twice.")

        num_networks = self.config["num_networks"]
        buffers = [serialize_tensor([self.input_buffer, self.output_buffer]) for _ in range(num_networks)]
        self.synchronization_queue.send_multipart([SyncCommands.REGISTER, pickle.dumps(buffers)])
        for _ in range(self.config["num_networks"]):
            network, self.identity = self.synchronization_queue.recv_multipart()

        self.response_queue.setsockopt(zmq.IDENTITY, self.identity)
        self.request_queue.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        self.response_queue.connect(relative_channel(ResponseManager.FRONTEND_CHANNEL, self.ipc_dir))

    def deregister(self):
        if not self.connected:
            raise ValueError("Cannot deregister a client that has not been registered")

        self.synchronization_queue.send_multipart([SyncCommands.DEREGISTER, self.identity])
        for _ in range(self.config["num_networks"]):
            self.synchronization_queue.recv_multipart()

        self.identity = None
        self.request_queue.disconnect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        self.response_queue.disconnect(relative_channel(ResponseManager.FRONTEND_CHANNEL, self.ipc_dir))

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deregister()

    def __del__(self):
        if self.identity is not None:
            self.synchronization_queue.send_multipart([SyncCommands.DEREGISTER, self.identity])

    def __call__(self, data):
        return self.predict(data)

    def _load_input_buffer(self, data, input_buffer=None) -> int:
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

    @property
    def connected(self):
        return self.identity is not None

    @property
    def predicting(self):
        return self.predict_size is not None

    def predict_inplace_async(self, size: int = None):
        assert self.connected, "Worker has tried to predict without registering first."
        assert not self.predicting, "Cannot launch two asynchronous prediction requests at once. " \
                                    "Must finish one before sending a new one."

        if size is None:
            size = self.batch_size

        self.predict_size = size
        self.request_queue.send_multipart([self.identity, serialize_int(size)])

    def predict_inplace(self, size: int = None):
        assert self.connected, "Worker has tried to predict without registering first."

        size = self.batch_size if size is None else size
        self.request_queue.send_multipart([self.identity, serialize_int(size)])

        self.response_queue.recv_multipart()
        return slice_buffer(self.output_buffer, 0, size)

    def predict(self, data):
        """ General prediction function for the client. Determines the correct type of prediction to make

        Parameters
        ----------
        data: A buffer of Numpy arrays, torch tensors, or integer size
            If data is an integer, then we predict on the current input buffer in place with the given size.
            If data is a Tensor, then we copy the tensor and predict on it
            If data is a numpy array, we convert the array into a tensor and predict on it.

        Returns
        -------
        A view into the output buffer as a torch tensor.
        """
        if isinstance(data, int):
            size = data
        else:
            size = self._load_input_buffer(data)

        # print(f"SIZE = {size}")
        return self.predict_inplace(size)

    def predict_async(self, data):
        assert not self.predicting, "Cannot launch two asynchronous prediction requests at once. " \
                                    "Must finish one before sending a new one."

        if isinstance(data, int):
            size = data
        else:
            size = self._load_input_buffer(data)

        self.predict_inplace_async(size)

    def receive_async(self):
        assert self.predicting, "Cannot receive a result until launching an asynchronous prediction request."

        self.response_queue.recv_multipart()

        predict_size = self.predict_size
        self.predict_size = None
        return slice_buffer(self.output_buffer, 0, predict_size)
