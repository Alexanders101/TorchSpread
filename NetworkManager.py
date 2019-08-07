import zmq
import torch
import itertools
import dill

from collections import Counter
from tempfile import TemporaryDirectory
from typing import Type, Union, Tuple, Dict, Optional, List

from NetworkWorker import NetworkWorker
from NetworkSynchronization import SynchronizationWorker, SynchronizationManager, SyncCommands
from NetworkBackend import forward_response
from utilities import deserialize_int, send_sigkill, serialize_tensor, relative_channel

from torch import multiprocessing, nn
from torch.multiprocessing import Process
# multiprocessing.set_sharing_strategy("file_system")
# multiprocessing.set_start_method('forkserver')


def round_robin_gpu_placement(num_networks=1):
    num_gpu = torch.cuda.device_count()
    if num_gpu < 1:
        raise ValueError("This machine has no GPUs installed. Cannot use gpu placement strategy.")

    gpus = ["cuda:{}".format(i) for i in range(num_gpu)]
    placement = {gpu: 0 for gpu in gpus}

    for _, gpu in zip(range(num_networks), itertools.cycle(gpus)):
        placement[gpu] += 1

    return placement


class RequestManager(Process):
    FRONTEND_CHANNEL = "ipc://request_frontend.ipc"
    BACKEND_CHANNEL = "ipc://request_backend.ipc"
    INTERCOMMUNICATION_CHANNEL = "ipc://network_interconnect.ipc"
    SENDER_CHANNEL = "inproc://sending_requests"

    # Command to shutdown all network workers
    SHUTDOWN = b'SHUTDOWN'

    def __init__(self, batch_size: int, network_identities: List[bytes], ipc_dir: str):
        super(RequestManager, self).__init__()

        self.batch_size = batch_size
        self.network_identities = network_identities
        self.num_networks = len(network_identities)
        self.ipc_dir = ipc_dir
        self.ready = multiprocessing.Event()

    @staticmethod
    def request_sender_thread(context: zmq.Context, backend, network_queue, ipc_dir):
        request_queue = context.socket(zmq.PULL)
        request_queue.connect(relative_channel(RequestManager.SENDER_CHANNEL, ipc_dir))

        while True:
            # Wait for a ready batch
            request = request_queue.recv_multipart()

            # Wait for an available network
            network = network_queue.recv()

            # Send the batched request
            backend.send(network, zmq.SNDMORE)
            backend.send_multipart(request)

    def _shutdown(self, backend: zmq.Socket):
        for network in self.network_identities:
            backend.send_multipart([network, NetworkWorker.SHUTDOWN])

    def run(self) -> None:
        context = zmq.Context()

        # Simple Single Pull channel to communicate available network workers
        network_queue = context.socket(zmq.PULL)
        network_queue.setsockopt(zmq.RCVBUF, 2 * self.num_networks)
        network_queue.set_hwm(1)
        network_queue.bind(relative_channel(self.INTERCOMMUNICATION_CHANNEL, self.ipc_dir))

        # Simple Push channel for sending the for batch for the network
        # request_queue = context.socket(zmq.PUSH)
        # request_queue.bind(relative_channel(self.SENDER_CHANNEL, self.ipc_dir))

        # Frontend channel to receive requests from clients
        frontend = context.socket(zmq.PULL)
        frontend.bind(relative_channel(self.FRONTEND_CHANNEL, self.ipc_dir))

        # Backend channel to send batches to networks
        backend = context.socket(zmq.ROUTER)
        backend.bind(relative_channel(self.BACKEND_CHANNEL, self.ipc_dir))

        # Poller for looking to see how many requests have been queue
        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)

        # Create the request sending thread
        # request_sender = Thread(target=self.request_sender_thread, args=[context, backend, network_queue])
        # request_sender.start()

        # We are ready for connections
        self.ready.set()

        # Main Loop
        current_batch = []
        carry_over = None

        while True:
            timeout = None
            current_size = 0
            current_batch.clear()

            # If we had data from the last batch, add it to the beginning of the current batch
            if carry_over is not None:
                current_batch.extend(carry_over)
                current_size = carry_over[1]
                timeout = 0
                carry_over = None

            # Wait for requests or see if we have any more requests
            num_requests = len(poller.poll(timeout))

            # Add all requests up to batch size to current prediction
            for _ in range(num_requests):
                request = frontend.recv_multipart(flags=zmq.NOBLOCK)

                # Shutdown event is just a size one request
                if len(request) == 1:
                    self._shutdown(backend)
                    return

                size = deserialize_int(request[1])

                # If we have reached the batch size and have more data, save the request for the next batch
                if current_size + size > self.batch_size:
                    carry_over = request
                    break

                # Otherwise, add the request to the batch
                else:
                    current_size += size
                    current_batch.extend(request)

            # request_queue.send_multipart(current_batch)
            # Wait for an available network
            network = network_queue.recv()

            # Send the request
            backend.send(network, zmq.SNDMORE)
            backend.send_multipart(current_batch)


class ResponseManager(Process):
    FRONTEND_CHANNEL = "ipc://response_frontend.ipc"
    BACKEND_CHANNEL = "ipc://response_backend.ipc"

    def __init__(self, num_networks: int, ipc_dir: str):
        super(ResponseManager, self).__init__(daemon=True)

        self.num_networks = num_networks
        self.ipc_dir = ipc_dir
        self.ready = multiprocessing.Event()

    def run(self) -> None:
        context = zmq.Context()

        # Frontend channel to receive requests from clients
        frontend = context.socket(zmq.DEALER)
        frontend.bind(relative_channel(self.FRONTEND_CHANNEL, self.ipc_dir))

        # Backend channel to send batches to networks
        backend = context.socket(zmq.PULL)
        backend.bind(relative_channel(self.BACKEND_CHANNEL, self.ipc_dir))

        # We are ready for connections
        self.ready.set()

        forward_response(backend, frontend)
        # while True:
        #     response = backend.recv_multipart()
        #     frontend.send_multipart(response)


class NetworkManager:
    def __init__(self,
                 network_class: Type[nn.Module],
                 input_shape: Union[Tuple, Dict[object, Tuple]],
                 input_type: Union[torch.dtype, Dict[object, torch.dtype]],
                 output_shape: Union[Tuple, Dict[object, Tuple]],
                 output_type: Union[torch.dtype, Dict[object, torch.dtype]],
                 placement: Optional[Dict[str, int]] = None,
                 local_network_device: str = 'cpu',
                 batch_size: int = 32,
                 network_args: List = None,
                 network_kwargs: Dict = None):
        # Create the mapping of where to place the worker networks
        self.placement = placement
        if placement is None:
            self.placement = {'cpu': 1}
        if isinstance(placement, list) or isinstance(placement, tuple):
            self.placement = dict(Counter(placement))

        self.ipc_dir_base = TemporaryDirectory()
        self.ipc_dir = self.ipc_dir_base.name

        # Network class and parameters
        self.network_class = network_class
        self.network_args = [] if network_args is None else network_args
        self.network_kwargs = {} if network_kwargs is None else network_kwargs

        # Buffer information
        self.input_shape = input_shape
        self.input_type = input_type
        self.output_shape = output_shape
        self.output_type = output_type

        # Predictor registration objects
        self.batch_size = batch_size
        self.local_network_device = local_network_device
        self.all_gpu = (all('cuda' in device for device in self.placement.keys()) and
                        'cuda' in local_network_device)

        # Create prediction networks
        self.networks = []
        self.network_identities = []
        self.num_networks = 0
        for device, count in self.placement.items():
            for _ in range(count):
                network = NetworkWorker(self.num_networks, device, self.network_config)
                self.networks.append(network)
                self.network_identities.append(network.network_identity)
                self.num_networks += 1

        # Create the management processes
        self.request_manager = RequestManager(self.batch_size, self.network_identities, self.ipc_dir)
        self.response_manager = ResponseManager(self.num_networks, self.ipc_dir)
        self.synchronization_manager = SynchronizationManager(self.ipc_dir)

        # Local network storage
        self._local_network = None
        self._state_dict = None

        # Local communication
        self.context = zmq.Context()
        self.synchronization_queue = self.context.socket(zmq.DEALER)

        self.started = False

    @property
    def network_config(self):
        return {
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "input_type": self.input_type,
            "output_shape": self.output_shape,
            "output_type": self.output_type,
            "network_class": self.network_class,
            "network_args": self.network_args,
            "network_kwargs": self.network_kwargs,
            "ipc_dir": self.ipc_dir,
            "request_channel": relative_channel(RequestManager.BACKEND_CHANNEL, self.ipc_dir),
            "response_channel": relative_channel(ResponseManager.BACKEND_CHANNEL, self.ipc_dir),
            "ready_channel": relative_channel(RequestManager.INTERCOMMUNICATION_CHANNEL, self.ipc_dir)
        }

    @property
    def client_config(self):
        return {
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "input_type": self.input_type,
            "output_shape": self.output_shape,
            "output_type": self.output_type,
            "num_networks": self.num_networks,
            "network_identities": self.network_identities,
            "all_gpu": self.all_gpu,
            "ipc_dir": self.ipc_dir
        }

    def synchronize(self):
        current_state_dict = self._local_network.state_dict()
        for key in current_state_dict:
            self._state_dict[key].data[:] = current_state_dict[key].data[:]

        self.synchronization_queue.send_multipart([SyncCommands.SYNC, b''])
        for _ in range(self.num_networks):
            self.synchronization_queue.recv_multipart()

    def shutdown(self):
        assert self.started

        context = zmq.Context()

        synchronization_queue = context.socket(zmq.DEALER)
        synchronization_queue.connect(relative_channel(SynchronizationWorker.SYNC_FRONTEND_CHANNEL, self.ipc_dir))

        request_queue = context.socket(zmq.PUSH)
        request_queue.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))

        synchronization_queue.send_multipart([SyncCommands.SHUTDOWN, b''])
        request_queue.send(RequestManager.SHUTDOWN)

        for network in self.networks:
            network.join()

        self.request_manager.join()
        send_sigkill(self.response_manager.pid)
        send_sigkill(self.synchronization_manager.pid)

    def start(self):
        assert not self.started

        self.request_manager.start()
        self.response_manager.start()
        self.synchronization_manager.start()

        print("Starting Request Manager")
        self.request_manager.ready.wait()

        print("Starting Response Manager")
        self.response_manager.ready.wait()

        print("Starting Synchronization Manager")
        self.synchronization_manager.ready.wait()

        for network in self.networks:
            network.start()

        for network, identity in zip(self.networks, self.network_identities):
            print("Starting Network {}".format(identity))
            network.ready.wait()

        print("Starting Local Network")
        self._local_network = self.network_class(*self.network_args, **self.network_kwargs)
        self._local_network = self._local_network.to(self.local_network_device)
        self._state_dict = self._local_network.state_dict()
        for parameter in self._state_dict.values():
            parameter.share_memory_()

        print("Synchronizing initial weights")
        self.synchronization_queue.connect(relative_channel(SynchronizationManager.SYNC_FRONTEND_CHANNEL, self.ipc_dir))
        self.synchronization_queue.send_multipart([SyncCommands.LOAD, serialize_tensor(self._state_dict)])
        for _ in range(self.num_networks):
            self.synchronization_queue.recv_multipart()

        self.synchronize()

        self.started = True
