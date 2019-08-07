import zmq

from typing import Dict
from utilities import make_buffer, serialize_tensor, serialize_int

from NetworkManager import ResponseManager, RequestManager
from NetworkSynchronization import SyncCommands, SynchronizationManager, relative_channel


class NetworkClient:
    def __init__(self, config: Dict, batch_size: int):
        assert batch_size <= config['batch_size'], "Client batch size is greater than manager batch size."

        self.config = config
        self.ipc_dir = config['ipc_dir']
        self.device = 'cuda' if config["all_gpu"] else 'shared'

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

    def register(self):
        buffers = [serialize_tensor([self.input_buffer, self.output_buffer]) for _ in range(self.config["num_networks"])]
        buffers = serialize_tensor(buffers)
        self.synchronization_queue.send_multipart([SyncCommands.REGISTER, buffers])
        for _ in range(self.config["num_networks"]):
            network, identity = self.synchronization_queue.recv_multipart()

        self.identity = identity
        self.response_queue.setsockopt(zmq.IDENTITY, identity)
        self.request_queue.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        self.response_queue.connect(relative_channel(ResponseManager.FRONTEND_CHANNEL, self.ipc_dir))

    def deregister(self):
        self.synchronization_queue.send_multipart([SyncCommands.DEREGISTER, self.identity])
        for _ in range(self.config["num_networks"]):
            self.synchronization_queue.recv_multipart()

        self.identity = None
        self.request_queue.disconnect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        self.response_queue.disconnect(relative_channel(ResponseManager.FRONTEND_CHANNEL, self.ipc_dir))

    @property
    def connected(self):
        return self.identity is not None

    def predict_inplace_async(self, size: int = None):
        assert self.connected, "Worker has tried to predict without registering first."

        if size is None:
            size = self.batch_size

        self.request_queue.send_multipart([self.identity, serialize_int(size)])

    def predict_inplace(self, size: int = None):
        self.predict_inplace_async(size)
        self.response_queue.recv()
        return self.output_buffer




