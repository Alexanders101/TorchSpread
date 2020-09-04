import numpy as np

import torch
import zmq
import msgpack

from torch import nn
from threading import Thread, Lock, Event

from .utilities import deserialize_tensor, relative_channel, mp_ctx


class SyncCommands:
    """ Namespace of commands that can be sent across the synchronization channels.
    """
    # Requests
    REGISTER = b'R'
    DEREGISTER = b'D'
    LOAD = b'L'
    SYNC = b'S'
    SHUTDOWN = b'K'

    # Responses
    SUCCESS = b'G'
    ERROR = b'E'


class SynchronizationWorker(Thread):
    # ZMQ Channels for communication
    SYNC_BACKEND_CHANNEL = "ipc://synchronization_backend.ipc"
    SYNC_FRONTEND_CHANNEL = "ipc://synchronization_frontend.ipc"

    def __init__(self, network: nn.Module, network_index: int, network_identity: bytes, ipc_dir: str,
                 context: zmq.Context = None):
        """ Asynchronous Thread responsible for adding clients to this networks registry and synchronizing weights
        with the training network.

        Parameters
        ----------
        network: torch.nn.Module
            The network on the main network worker.
        network_index: int
            The index of the parent network worker.
        network_identity: bytes
            The identity of this network worker.
        ipc_dir: str
            Prefix for IPC channels.
        context: zmq.Context
            ZMQ context used in the main network worker.
        """
        super(SynchronizationWorker, self).__init__()

        self.context = zmq.Context() if context is None else context
        self.ipc_dir = ipc_dir

        self.network = network
        self.network_index = network_index
        self.network_identity = network_identity
        self.network_lock = Lock()
        self.ready = Event()

        self.state_dict = {}
        self.input_buffers = {}
        self.output_buffers = {}

    def _register(self, client, input_buffer, output_buffer):
        self.input_buffers[client] = input_buffer
        self.output_buffers[client] = output_buffer

    def _deregister(self, client):
        del self.input_buffers[client]
        del self.output_buffers[client]

    def _load(self, state_dict):
        self.state_dict = state_dict

    def _cleanup(self):
        del self.input_buffers
        del self.output_buffers
        del self.state_dict

    def run(self) -> None:
        request_queue = self.context.socket(zmq.SUB)
        request_queue.setsockopt(zmq.SUBSCRIBE, b'')
        request_queue.connect(relative_channel(self.SYNC_BACKEND_CHANNEL, self.ipc_dir))

        response_queue = self.context.socket(zmq.DEALER)
        response_queue.setsockopt(zmq.IDENTITY, self.network_identity)
        response_queue.connect(relative_channel(self.SYNC_FRONTEND_CHANNEL, self.ipc_dir))

        self.ready.set()
        while True:
            identity, command, data = request_queue.recv_multipart()

            # Worker network synchronization
            if command == SyncCommands.SYNC:
                with self.network_lock:
                    self.network.load_state_dict(self.state_dict)
            elif command == SyncCommands.LOAD:
                self.state_dict = deserialize_tensor(msgpack.loads(data)[self.network_index])

            # Local client registration
            elif command == SyncCommands.REGISTER:
                data = deserialize_tensor(msgpack.loads(data)[self.network_index])
                self._register(identity, *data)
            elif command == SyncCommands.DEREGISTER:
                self._deregister(data)

            # Kill command
            elif command == SyncCommands.SHUTDOWN:
                self._cleanup()
                break

            response_queue.send_multipart([SyncCommands.SUCCESS, identity])


class SynchronizationManager(mp_ctx.Process):
    SYNC_BACKEND_CHANNEL = SynchronizationWorker.SYNC_BACKEND_CHANNEL
    SYNC_FRONTEND_CHANNEL = SynchronizationWorker.SYNC_FRONTEND_CHANNEL

    def __init__(self, ipc_dir: str):
        super(SynchronizationManager, self).__init__(daemon=True)
        self.ipc_dir = ipc_dir
        self.ready = mp_ctx.Event()

    def run(self) -> None:
        context = zmq.Context()

        # Main Publisher for the synchronization
        synchronization_backend = context.socket(zmq.PUB)
        synchronization_backend.bind(relative_channel(self.SYNC_BACKEND_CHANNEL, self.ipc_dir))

        # Main Router for Response channel to manage connections
        synchronization_frontend = context.socket(zmq.ROUTER)
        synchronization_frontend.bind(relative_channel(self.SYNC_FRONTEND_CHANNEL, self.ipc_dir))

        # We are ready for connections
        self.ready.set()

        while True:
            identity, command, data = synchronization_frontend.recv_multipart()

            if command == SyncCommands.SUCCESS:
                synchronization_frontend.send_multipart([data, identity, data])
            else:
                synchronization_backend.send_multipart([identity, command, data])
