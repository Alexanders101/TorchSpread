import zmq
import pickle

from typing import Dict
from utilities import make_buffer, serialize_tensor, serialize_int, load_buffer, optional

from NetworkManager import ResponseManager, RequestManager
from NetworkSynchronization import SyncCommands, SynchronizationManager, relative_channel


class NetworkClient:
    def __init__(self, config: Dict, batch_size: int, device=None):
        assert batch_size <= config['batch_size'], "Client batch size is greater than manager batch size."

        self.config = config
        self.ipc_dir = config['ipc_dir']

        self.device = optional(device, 'shared')
        if 'cpu' in self.device:
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

        buffers = [serialize_tensor([self.input_buffer, self.output_buffer]) for _ in range(self.config["num_networks"])]
        self.synchronization_queue.send_multipart([SyncCommands.REGISTER, pickle.dumps(buffers)])
        for _ in range(self.config["num_networks"]):
            network, identity = self.synchronization_queue.recv_multipart()

        self.identity = identity
        self.response_queue.setsockopt(zmq.IDENTITY, identity)
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

    def receive_async(self):
        assert self.predicting, "Cannot receive a result until launching an asynchronous prediction request."

        self.response_queue.recv_multipart()
        self.predict_size = None
        return self.output_buffer[:self.predict_size]

    def predict_inplace(self, size: int = None):
        assert self.connected, "Worker has tried to predict without registering first."

        size = self.batch_size if size is None else size
        self.request_queue.send_multipart([self.identity, serialize_int(size)])

        self.response_queue.recv_multipart()
        return self.output_buffer[:size]

    def predict_tensor(self, data):
        if isinstance(data, dict):
            size = next(iter(data.values())).size(0)
        elif isinstance(data, list):
            size = data[0].size(0)
        else:
            size = data.size(0)

        load_buffer(self.input_buffer, data, size, 0)
        return self.predict_inplace(size)



