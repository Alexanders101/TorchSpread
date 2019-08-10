import zmq
import torch
import pickle
import itertools

from contextlib import contextmanager
from collections import Counter
from tempfile import TemporaryDirectory
from typing import Type, Union, Tuple, Dict, Optional, List

from .NetworkWorker import NetworkWorker
from .NetworkSynchronization import SynchronizationManager, SyncCommands
from .utilities import deserialize_int, send_sigkill, serialize_tensor, relative_channel, optional

from torch import multiprocessing

mp_ctx = multiprocessing.get_context('forkserver')
Process = mp_ctx.Process


class PlacementStrategy:
    """
    Namespace containing various automated placement strategies for
    placing networks onto many-gpu or many-cpu machines.
    """
    @staticmethod
    def _get_available_gpus():
        num_gpu = torch.cuda.device_count()
        if num_gpu < 1:
            raise ValueError("This machine has no GPUs installed. Cannot use gpu placement strategy.")
        return ["cuda:{}".format(i) for i in range(num_gpu)]

    @staticmethod
    def round_robin_gpu_placement(num_networks: int = 1):
        """ Place networks sequentially across all available GPUs.

        Parameters
        ----------
        num_networks: int
            Total number of networks to place.
        """
        gpus = PlacementStrategy._get_available_gpus()
        placement = {gpu: 0 for gpu in gpus}

        for _, gpu in zip(range(num_networks), itertools.cycle(gpus)):
            placement[gpu] += 1

        return placement

    @staticmethod
    def uniform_gpu_placement(networks_per_gpu=1):
        """ Place an equal number of networks on all available GPU.
        Parameters
        ----------
        networks_per_gpu: int
            Number of networks to place on each GPU.
            Total number of networks will be #GPU * networks_per_gpu
        """
        return {gpu: networks_per_gpu for gpu in PlacementStrategy._get_available_gpus()}

    @staticmethod
    def uniform_cpu_placement(num_networks: int = 1):
        """ Place a number of networks on only cpus.

        Parameters
        ----------
        num_networks: int
            Total number of networks to place.
        """
        return {'cpu': num_networks}


class NetworkManager:
    def __init__(self,
                 input_shape: Union[Tuple, List[Tuple], Dict[object, Tuple]],
                 input_type: Union[torch.dtype, List[torch.dtype], Dict[object, torch.dtype]],
                 output_shape: Union[Tuple, List[Tuple], Dict[object, Tuple]],
                 output_type: Union[torch.dtype, List[torch.dtype], Dict[object, torch.dtype]],
                 batch_size: int,
                 network_class: Type,
                 network_args: List = None,
                 network_kwargs: Dict = None,
                 placement: Optional[Union[List[str], Dict[str, int]]] = None,
                 training_placement: Optional[str] = None,
                 num_worker_buffers: int = 2):
        """ Primary manager class for creating a distributed prediction cluster.

        Notes
        -----
        For the input and output shapes / types, we allow any combination of shapes, list of shapes,
        or dictionaries of shapes. I.E. if your network has a single input, then the input shape will simply
        be the tuple of the shape. If you have a list of inputs, then the shape will be
        a list of tuple shapes. If you have a dictionary of inputs, then the shape wil be a
        named dictionary of tuple shapes.

        Please ensure that your forward function takes in a single argument with the same structure
        as your input shape / type and returns a single output with the same structure as the
        output shape / type.

        Parameters
        ----------
        input_shape: {Tuple, List[Tuple], Dict[Tuple]}
            The shape(s) of the input to your network's forward function.
        input_type: {Tuple, List[Tuple], Dict[Tuple]}
            The type(s) of the input to your network's forward function.
        output_shape: {Tuple, List[Tuple], Dict[Tuple]}
            The shape(s) of the output to your network's forward function.
        output_type: {Tuple, List[Tuple], Dict[Tuple]}
            The type(s) of the output to your network's forward function.
        batch_size: int
            The maximum batch size for the worker networks. Note that the networks can
            predict on samples less than the batch size, we simply need to create the input buffers
            with a given batch size.
        network_class: Subclass of torch.nn.Module
            Your network class. Please ensure the forward function has a single input and output.
        network_args: List
            List of arguments to pass to the network constructor.
        network_kwargs: Dict
            Dictionary of keyword arguments to pass to the network constructor.
        placement: Dict[str, int] or List[str]
            Dictionary mapping a device name to the number of networks on that device or
            List containing the devices for each network.
            Defaults to {'cpu': 1}
        training_placement: str
            Device name to place the local training network on. Defaults to 'cuda' if
            all of the networks are on the gpu or 'cpu' otherwise.
        num_worker_buffers: int
            Number of asynchronous buffers allocated for each worker. More buffers allows for more data to be
            copying at the same time. However, these buffers take up space. This is most useful to increase when
            you have a relatively small network with a lot of inputs, meaning you spend most of your time
            copying data into and out of the network. Default is 2.
        """
        # Temporary directory holding all of the communication objects
        self._ipc_dir_base = TemporaryDirectory()
        self.ipc_dir = self._ipc_dir_base.name

        # Create the mapping of where to place the worker networks
        self.batch_size = batch_size
        self.num_worker_buffers = num_worker_buffers
        self.placement = self._create_placement(placement)
        self.training_placement = self._create_training_placement(training_placement)

        # Network class and parameters
        self.network_class = network_class
        self.network_args = optional(network_args, default=[])
        self.network_kwargs = optional(network_kwargs, default={})

        # Buffer information
        self.input_shape = input_shape
        self.input_type = input_type
        self.output_shape = output_shape
        self.output_type = output_type

        # Create prediction networks
        self.networks = []
        self.network_identities = []
        self.num_networks = 0
        self._create_networks()

        # Create the management processes
        self.request_manager = RequestManager(self.batch_size, self.num_networks, self.network_identities, self.ipc_dir)
        self.response_manager = ResponseManager(self.num_networks, self.ipc_dir)
        self.synchronization_manager = SynchronizationManager(self.ipc_dir)

        # Local network storage
        self._local_network = None
        self._state_dict = None
        self._network_lock = mp_ctx.Lock()

        # Local communication
        self.context = zmq.Context()
        self.synchronization_queue = self.context.socket(zmq.DEALER)

        # State objects
        self.started = False
        self.killed = False

    @staticmethod
    def _create_placement(placement):
        if placement is None:
            return {'cpu': 1}
        elif isinstance(placement, list) or isinstance(placement, tuple):
            return dict(Counter(placement))
        else:
            return placement

    def _create_training_placement(self, local_placement):
        default_placement = 'cuda' if all(('cuda' in device) for device in self.placement) else 'cpu'
        return optional(local_placement, default_placement)

    def _create_networks(self):
        for device, count in self.placement.items():
            for _ in range(count):
                network = NetworkWorker(self.num_networks, device, self.network_config)
                self.networks.append(network)
                self.network_identities.extend(network.request_worker_identities)
                self.num_networks += 1

    def __del__(self):
        if self.started and not self.killed:
            self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

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
            "num_worker_buffers": self.num_worker_buffers,
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
            "ipc_dir": self.ipc_dir
        }

    def _check_started(self):
        if not self.started:
            raise AssertionError("Manager must be started before performing commands.")

    def _send_synchronization_command(self, command: bytes, data: bytes = b''):
        self.synchronization_queue.send_multipart([command, data])
        for _ in range(self.num_networks):
            self.synchronization_queue.recv_multipart()

    def synchronize(self):
        """ Push the current training network weights to the worker networks. """
        self._check_started()

        # Update the shared state dict
        training_weights = self._local_network.state_dict()
        shared_weights = self._state_dict
        for key in shared_weights:
            shared_weights[key].copy_(training_weights[key])

        # Synchronize Workers
        self._send_synchronization_command(SyncCommands.SYNC)

    @property
    def training_parameters(self):
        """ Get the parameters of the training network.

        This is useful for creating the optimizer or visualization.
        """
        self._check_started()
        return self._local_network.parameters()

    @property
    def state_dict(self):
        """ Get the current state dict of the training network"""
        self._check_started()
        return self._local_network.state_dict()

    @property
    @contextmanager
    def training_network(self):
        """ Get a synchronized context for the current training network.

        This is meant to be used in a with block. After the context has been closed
        it will automatically synchronize the network.
        """
        self._check_started()
        with self._network_lock:
            yield self._local_network
            self.synchronize()

    def load_state_dict(self, state_dict, strict=True):
        """ Load a state dict into the training network.

        This is the synchronous version of the state dict loading which will
        automatically synchronize the weights of the worker networks after loading.
        """
        self._check_started()
        with self._network_lock:
            self._local_network.load_state_dict(state_dict, strict=strict)
            self.synchronize()

    @property
    def unsynchronized_training_network(self):
        """ Access the raw training network with no automatic synchronization. """
        self._check_started()
        return self._local_network

    def unsynchronized_load_state_dict(self, state_dict, strict=True):
        """ Load the state dict into the training network.

        This is th unsynchronized version of state dict loading. The worker networks will not
        be automatically synchronized, which means their weights will not update!
        """
        self._check_started()
        self._local_network.load_state_dict(state_dict, strict=strict)

    def start(self, verbose: bool = True):
        """ Start the network manager and all of the worker networks.

        Parameters
        ----------
        verbose: bool
            Whether or not to print the phase of launch.
        """
        assert not self.started, "Manager should not be started twice"
        printer = print if verbose else (lambda x: x)

        self.request_manager.start()
        self.response_manager.start()
        self.synchronization_manager.start()

        printer("Starting Request Manager")
        self.request_manager.ready.wait()

        printer("Starting Response Manager")
        self.response_manager.ready.wait()

        printer("Starting Synchronization Manager")
        self.synchronization_manager.ready.wait()

        for network in self.networks:
            network.start()

        for network in self.networks:
            printer("Starting Network {} on {}".format(network.identity, network.device))
            network.ready.wait()

        printer("Starting Local Network")
        self._local_network = self.network_class(*self.network_args, **self.network_kwargs)
        self._local_network = self._local_network.to(self.training_placement)
        self._state_dict = self._local_network.state_dict()
        for parameter in self._state_dict.values():
            parameter.share_memory_()

        printer("Synchronizing initial weights")
        self.synchronization_queue.connect(relative_channel(SynchronizationManager.SYNC_FRONTEND_CHANNEL, self.ipc_dir))
        state_buffers = [serialize_tensor(self._state_dict) for _ in range(self.num_networks)]
        self._send_synchronization_command(SyncCommands.LOAD, pickle.dumps(state_buffers))

        self.started = True
        self.synchronize()

    def shutdown(self):
        self._check_started()
        assert not self.killed, "Manager should not be shutdown twice."

        # Shutdown the synchronization channels
        self.synchronization_queue.send_multipart([SyncCommands.SHUTDOWN, b''])

        # Create a local request queue to shutdown request channels
        request_queue = self.context.socket(zmq.PUSH)
        request_queue.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        request_queue.send(RequestManager.SHUTDOWN)

        # Wait for the workers to finish their cleanup
        self.request_manager.join()
        for network in self.networks:
            network.join()

        # Manually kill the other two managers. They dont hold any locks or files, so this is fine.
        send_sigkill(self.response_manager.pid)
        send_sigkill(self.synchronization_manager.pid)

        # We can only be killed once
        self.killed = True


class RequestManager(Process):
    FRONTEND_CHANNEL = "ipc://request_frontend.ipc"
    BACKEND_CHANNEL = "ipc://request_backend.ipc"
    INTERCOMMUNICATION_CHANNEL = "ipc://network_interconnect.ipc"
    SENDER_CHANNEL = "inproc://sending_requests"

    # Command to shutdown all network workers
    SHUTDOWN = b'SHUTDOWN'

    def __init__(self, batch_size: int, num_networks: int, network_identities: List[bytes], ipc_dir: str):
        super(RequestManager, self).__init__()

        self.batch_size = batch_size
        self.network_identities = network_identities
        self.num_networks = num_networks
        self.ipc_dir = ipc_dir
        self.ready = mp_ctx.Event()

    def _shutdown(self, backend: zmq.Socket):
        for network in self.network_identities:
            backend.send_multipart([network, pickle.dumps([NetworkWorker.SHUTDOWN])])

    def run(self) -> None:
        context = zmq.Context()

        # Simple Single Pull channel to communicate available network workers
        # NetworkWorker -> Ready Command -> [RequestManager]
        network_queue = context.socket(zmq.PULL)
        network_queue.bind(relative_channel(self.INTERCOMMUNICATION_CHANNEL, self.ipc_dir))

        # Frontend channel to receive requests from clients
        # NetworkClient -> (identity, request_size) -> [RequestManager]
        frontend = context.socket(zmq.PULL)
        frontend.bind(relative_channel(self.FRONTEND_CHANNEL, self.ipc_dir))

        # Backend channel to send batches to networks
        # [RequestManager] -> Batch -> NetworkWorkerRequester
        backend = context.socket(zmq.ROUTER)
        backend.bind(relative_channel(self.BACKEND_CHANNEL, self.ipc_dir))

        # Poller for looking to see how many requests have been queue
        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)

        # We are ready for connections
        self.ready.set()

        # Main Loop
        current_batch = []
        carry_over = None

        while True:
            flags = 0
            current_size = 0
            current_batch.clear()

            # If we had data from the last batch, add it to the beginning of the current batch
            if carry_over is not None:
                flags = zmq.NOBLOCK
                current_size = deserialize_int(carry_over[1])
                current_batch.extend(carry_over)
                carry_over = None

            # Add all requests up to batch size to current prediction
            while True:
                try:
                    request = frontend.recv_multipart(flags=flags)
                    flags = zmq.NOBLOCK
                except zmq.ZMQError:
                    break

                # Shutdown event is just a size one request
                if len(request) == 1:
                    self._shutdown(backend)
                    return

                size = deserialize_int(request[1])
                new_size = current_size + size

                # If we have reached the batch size and have more data, save the request for the next batch
                if new_size > self.batch_size:
                    carry_over = request
                    break

                # Otherwise, add the request to the batch
                else:
                    current_size = new_size
                    current_batch.extend(request)

            # Wait for an available network
            network = network_queue.recv()

            # Send the request
            backend.send_multipart([network, pickle.dumps(current_batch)])


class ResponseManager(Process):
    FRONTEND_CHANNEL = "ipc://response_frontend.ipc"
    BACKEND_CHANNEL = "ipc://response_backend.ipc"

    def __init__(self, num_networks: int, ipc_dir: str):
        super(ResponseManager, self).__init__(daemon=True)

        self.num_networks = num_networks
        self.ipc_dir = ipc_dir
        self.ready = mp_ctx.Event()

    def run(self) -> None:
        context = zmq.Context()

        # Frontend channel to send responses to clients
        # [ResponseManager] -> Done Command -> NetworkClient
        frontend = context.socket(zmq.ROUTER)
        frontend.bind(relative_channel(self.FRONTEND_CHANNEL, self.ipc_dir))

        # Backend channel to send batches to networks
        # NetworkWorkerResponder -> Done Command -> [ResponseManager]
        backend = context.socket(zmq.PULL)
        backend.bind(relative_channel(self.BACKEND_CHANNEL, self.ipc_dir))

        # We are ready for connections
        self.ready.set()

        while True:
            response = backend.recv_multipart()
            frontend.send_multipart(response)
