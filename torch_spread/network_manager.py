import numpy as np

import zmq
import zmq.devices
import msgpack

from contextlib import contextmanager
from collections import Counter
from tempfile import TemporaryDirectory
from typing import Type, Union, Dict, Optional, List, Any, Callable

from torch import nn

from .network_worker import NetworkWorker
from .manager_tools import TrainingWrapper
from .buffer_tools import make_buffer_shape_type
from .network_synchronization import SynchronizationManager, SyncCommands

from .utilities import mp_ctx
from .utilities import serialize_buffer, deserialize_buffer, ShapeBufferType, DtypeBufferType
from .utilities import deserialize_int, send_sigkill, serialize_tensor, relative_channel, optional


class NetworkManager:
    def __init__(self,
                 input_shape: Union[int, ShapeBufferType],
                 input_type: Optional[DtypeBufferType],
                 output_shape: Union[int, ShapeBufferType],
                 output_type: Optional[DtypeBufferType],
                 batch_size: int,
                 network_class: Type[nn.Module],
                 network_args: List = None,
                 network_kwargs: Dict = None,
                 placement: Optional[Union[List[str], Dict[str, int]]] = None,
                 training_placement: Optional[str] = None,
                 training_wrapper: Optional[TrainingWrapper] = None,
                 remote_manager: Optional[Union[int, str]] = None,
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
        input_shape: int or BufferShape
            The shape(s) of the input to your network's forward function.
            An int will be assumed as single dimensional tensor with a given length.
        input_type: Optional[{Tuple, List[Tuple], Dict[Tuple]}]
            The type(s) of the input to your network's forward function.
            `None` will create a buffer with the same structure as input_shape and torch.float32 for all tensors.
        output_shape: int or BufferShape
            The shape(s) of the output to your network's forward function.
            An int will be assumed as single dimensional tensor with a given length.
        output_type: {Tuple, List[Tuple], Dict[Tuple]}
            The type(s) of the output to your network's forward function.
            `None` will create a buffer with the same structure as output_shape and torch.float32 for all tensors.
        batch_size: int
            The maximum batch size to be fed the networks. Note that the networks can
            predict on samples less than the batch size, we simply need to create the input buffers
            with a given batch size. If a client wants to predict on a larger set of samples,
            then they must manually perform batching to match the network batch size.
        network_class: Subclass of torch.nn.Module
            Your network class. Please ensure the forward function has a single input and output. Although these
            inputs and output can be any buffer structure. Furthermore, the first parameter in the __init__
            function has to be 'worker', a variable stating if it is a worker network or the training network.
            You can use NetworkTools.SpreadModule to ensure compatibility.
        network_args: List
            List of arguments to pass to the network constructor.
        network_kwargs: Dict
            Dictionary of keyword arguments to pass to the network constructor.
        placement: Dict[str, int] or List[str]
            Dictionary mapping a device name to the number of networks on that device or
            List containing the devices for each network. You may also use NetworkTools.PlacementStrategy
            to generate these placements with common patterns. Defaults to {'cpu': 1}.
        training_placement: str
            Device name to place the local training network on. Defaults to 'cuda' if
            all of the networks are on the gpu or 'cpu' otherwise.
        training_wrapper: function mapping (nn.Module, ) -> nn.Module
            An extra wrapping function applied to the training network (but not the worker networks).
            One example use-case of this would be to add data-parallelism to the training network.
        remote_manager: int or str, optional
            If this parameter is provided, this manager will start a remote manager so that remote clients can connect
            to this machine. If an int is provided, it is treated as a port number and the manager will bind on
            all hostnames. Alternatively, you may provide a string of the format 'hostname:port'.
        num_worker_buffers: int
            Number of asynchronous buffers allocated for each worker. More buffers allows for more data to be
            copying at the same time. However, these buffers take up space. This is most useful to increase when
            you have a relatively small network with a lot of inputs, meaning you spend most of your time
            copying data into and out of the network. Default is 2 buffers.
        """
        # Temporary directory holding all of the communication objects
        self._ipc_dir_base = TemporaryDirectory()
        self.ipc_dir = self._ipc_dir_base.name

        # Create the mapping of where to place the worker networks
        self.batch_size = batch_size
        self.num_worker_buffers = num_worker_buffers
        self.placement = self._create_placement(placement)
        self.training_placement = self._create_training_placement(training_placement)
        self.training_wrapper = optional(training_wrapper, TrainingWrapper())

        # Network class and parameters
        self.network_class = network_class
        self.network_args = optional(network_args, default=[])
        self.network_kwargs = optional(network_kwargs, default={})

        # Buffer information
        self.input_shape, self.input_type = make_buffer_shape_type(input_shape, input_type)
        self.output_shape, self.output_type = make_buffer_shape_type(output_shape, output_type)

        # Create prediction networks
        self.networks: List[NetworkWorker] = []
        self.network_identities: List[bytes] = []
        self.num_networks: int = 0
        self._create_networks()

        # Create the management processes
        self.request_manager = RequestManager(self.batch_size, self.num_networks, self.network_identities, self.ipc_dir)
        self.frontend_manager = FrontendManager(self.num_networks, self.ipc_dir)
        self.synchronization_manager = SynchronizationManager(self.ipc_dir)

        # Remote manager must be started during the start procedure instead of the init method
        self.remote_manager_config = remote_manager
        self.remote_manager = None

        # Local network storage
        self._local_network: Optional[nn.Module] = None
        self._state_dict: Optional[dict] = None
        self._network_lock = mp_ctx.Lock()

        # Local communication
        self.context = zmq.Context()
        self.synchronization_queue = self.context.socket(zmq.DEALER)
        self.synchronization_poller = zmq.Poller()

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

    def _check_started(self):
        if not self.started:
            raise AssertionError("Manager must be started before performing commands.")

    def _send_synchronization_command(self, command: bytes, data: bytes = b'', timeout: Optional[int] = None):
        self.synchronization_queue.send_multipart([command, data])

        sockets = dict(self.synchronization_poller.poll(timeout))
        if self.synchronization_queue not in sockets:
            raise TimeoutError("Synchronization Timed Out")

        for _ in range(self.num_networks):
            self.synchronization_queue.recv_multipart()

    def __del__(self):
        if self.started and not self.killed:
            self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def network_config(self) -> Dict[str, Any]:
        """ This configuration dictionary is passed to network workers on the backend. """
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
            "response_channel": relative_channel(FrontendManager.BACKEND_CHANNEL, self.ipc_dir),
            "ready_channel": relative_channel(RequestManager.INTERCOMMUNICATION_CHANNEL, self.ipc_dir)
        }

    @property
    def client_config(self) -> Dict[str, Any]:
        """ This configuration dictionary is passed to any local clients. """
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

    def synchronize(self, timeout: Optional[int] = None) -> None:
        """ Push the current training network weights to the worker networks.

        Notes
        -----
        This will update the remove network weights synchronously. Blocks until all of the networks
        indicate that they are ready.
        """
        self._check_started()

        # Update the shared state dict
        training_weights = self.training_wrapper.wrap_state_dict(self._local_network.state_dict())
        shared_weights = self._state_dict
        for key in shared_weights:
            shared_weights[key].copy_(training_weights[key])

        # Synchronize Workers
        self._send_synchronization_command(SyncCommands.SYNC, timeout=timeout)

    @property
    def training_parameters(self):
        """ Get the parameters of the training network.

        This is useful for creating the optimizer or visualization.
        """
        self._check_started()
        return self._local_network.parameters()

    @property
    def state_dict(self):
        """ Get the current state dict of the training network. """
        self._check_started()
        return self._local_network.state_dict()

    @property
    def serialized_state_dict(self):
        """ Get a serialized version of the state dict to save or send across a network. """
        return serialize_buffer(self.state_dict)

    @property
    def compressed_state_dict(self):
        """ Get a compressed serialized state dict. Useful for large models. """
        return serialize_buffer(self.state_dict, compress=3)

    @property
    @contextmanager
    def training_network(self, synchronization_timeout: Optional[int] = 1000):
        """ Get a synchronized context for the current training network.

        This is meant to be used in a context 'with' block. After the context is closed
        it will automatically synchronize the network.

        Parameters
        ----------
        synchronization_timeout: int, optional
            Timeout, in ms, for the synchronization command after returning for the with block.
            Useful to ensure that the worker networks are still alive.
        """
        self._check_started()
        with self._network_lock:
            yield self._local_network
            self.synchronize(synchronization_timeout)

    def load_state_dict(self, state_dict, synchronization_timeout: Optional[int] = 1000, strict: bool = True):
        """ Load a state dict into the training network.

        This is the synchronous version of the state dict loading which will
        automatically synchronize the weights of the worker networks after loading.
        """
        self._check_started()
        with self._network_lock:
            self._local_network.load_state_dict(state_dict, strict=strict)
            self.synchronize(synchronization_timeout)

    def deserialize_state_dict(self, serialized_state_dict, strict=True):
        """ Load a serialized state dict into the training network.

        This is the synchronous version of the state dict loading which will automatically synchronize
        the weights of the worker networks after loading. This will work with either the regular serialized
        state dict or the compressed state dict.

        """
        state_dict = deserialize_buffer(serialized_state_dict)
        self.load_state_dict(state_dict, strict)

    @property
    def unsynchronized_training_network(self) -> nn.Module:
        """ Access the raw training network with no automatic synchronization. """
        self._check_started()
        return self._local_network

    def unsynchronized_load_state_dict(self, state_dict, strict=True):
        """ Load the state dict into the training network.

        This is the unsynchronized version of state dict loading. The worker networks will not
        be automatically synchronized. This means that while the training network will receive
        the new weights, the worker network weights will not update!
        """
        self._check_started()
        self._local_network.load_state_dict(state_dict, strict=strict)

    def start(self, verbose: bool = True) -> None:
        """ Start the network manager and all of the worker networks.

        Parameters
        ----------
        verbose: bool
            Whether or not to print the phases of launch.
        """
        assert not self.started, "Manager should not be started twice"
        printer: Callable[[str], None] = print if verbose else (lambda x: x)

        self.request_manager.start()
        self.frontend_manager.start()
        self.synchronization_manager.start()

        printer("Starting Request Manager")
        self.request_manager.ready.wait()

        printer("Starting Response Manager")
        self.frontend_manager.ready.wait()

        printer("Starting Synchronization Manager")
        self.synchronization_manager.ready.wait()

        for network in self.networks:
            network.start()

        for network in self.networks:
            printer("Starting Network {} on {}".format(network.identity, network.device))
            network.ready.wait()

        printer("Creating Local Network")
        self._local_network = self.network_class(False, *self.network_args, **self.network_kwargs)
        self._local_network = self._local_network.to(self.training_placement)
        self._local_network = self.training_wrapper.wrap_network(self._local_network)

        printer("Synchronizing initial weights")
        self._state_dict = self.training_wrapper.wrap_state_dict(self._local_network.state_dict())
        for key, parameter in self._state_dict.items():
            self._state_dict[key] = parameter.clone().share_memory_()

        self.synchronization_queue.connect(relative_channel(SynchronizationManager.SYNC_FRONTEND_CHANNEL, self.ipc_dir))
        self.synchronization_poller.register(self.synchronization_queue, zmq.POLLIN)
        state_buffers = [serialize_tensor(self._state_dict) for _ in range(self.num_networks)]
        self._send_synchronization_command(SyncCommands.LOAD, msgpack.dumps(state_buffers))

        if self.remote_manager_config is not None:
            # This import has to be included here because otherwise it creates a circular reference
            from .remote_manager import RemoteManager
            if isinstance(self.remote_manager_config, int):
                hostname = "*"
                port = self.remote_manager_config
            else:
                hostname, port = self.remote_manager_config.split(":")

            printer(f"Starting remote manager on {hostname}:{port}")
            self.remote_manager = RemoteManager(self.client_config, int(port), hostname)
            self.remote_manager.start()
            self.remote_manager.ready.wait()

        self.started = True
        self.synchronize()

    def shutdown(self) -> None:
        """ Shutdown the cluster and all associated networks. """
        self._check_started()
        assert not self.killed, "Manager should not be shutdown twice."

        if self.remote_manager is not None:
            self.remote_manager.shutdown(self.context)

        # Shutdown the synchronization channels
        self.synchronization_queue.send_multipart([SyncCommands.SHUTDOWN, b''])

        # Create a local request queue to shutdown request channels
        request_queue = self.context.socket(zmq.PUSH)
        request_queue.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))
        request_queue.send(RequestManager.SHUTDOWN)

        # Manually kill the other two managers. They dont hold any locks or files, so this is fine.
        try:
            send_sigkill(self.frontend_manager.pid)
            send_sigkill(self.synchronization_manager.pid)
        except ProcessLookupError:
            pass

        # Wait for the workers to finish their cleanup
        self.request_manager.join(timeout=1)
        for network in self.networks:
            network.join(timeout=1)

        if self.remote_manager is not None:
            self.remote_manager.join(timeout=1)

        # We can only be killed once
        self.killed = True


class RequestManager(mp_ctx.Process):
    """ Process to batch all incoming prediction requests and load balance the networks.

    Notes
    -----
    This process does not perform the in-memory batches. Those are performed by the workers so that it can be done in
    parallel. This class simply assigns requests to batches and sends them to the network workers to perform batching.
    """
    FRONTEND_CHANNEL = "ipc://request_frontend.ipc"
    BACKEND_CHANNEL = "ipc://request_backend.ipc"
    INTERCOMMUNICATION_CHANNEL = "ipc://network_interconnect.ipc"

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
            backend.send_multipart([network, msgpack.dumps([NetworkWorker.SHUTDOWN])])

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
                # In python, trying and failing is faster than asking beforehand...
                try:
                    size = deserialize_int(request[1])
                    new_size = current_size + size
                except IndexError:
                    self._shutdown(backend)
                    return

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
            backend.send_multipart([network, msgpack.dumps(current_batch)])


class FrontendManager(mp_ctx.Process):
    """ Main frontend class for interacting with clients. This is essentially a high speed router.

    This class just forwards requests / responses between the backend and their associated clients. """

    FRONTEND_CHANNEL = "ipc://response_frontend.ipc"
    BACKEND_CHANNEL = "ipc://response_backend.ipc"

    def __init__(self, num_networks: int, ipc_dir: str):
        super(FrontendManager, self).__init__(daemon=True)

        self.num_networks = num_networks
        self.ipc_dir = ipc_dir
        self.ready = mp_ctx.Event()

    def run(self) -> None:
        context = zmq.Context()

        # Frontend channel to send and receive requests from clients
        # NetworkClient -> Request Command -> [FrontendManager]
        # [FrontendManager] -> Done Command -> NetworkClient
        frontend = context.socket(zmq.ROUTER)
        frontend.bind(relative_channel(self.FRONTEND_CHANNEL, self.ipc_dir))

        # Backend channel to receive result batches from networks
        # ResponseWorker -> Done Command -> [FrontendManager]
        response_backend = context.socket(zmq.PULL)
        response_backend.bind(relative_channel(self.BACKEND_CHANNEL, self.ipc_dir))

        # Backend channel to send requests to request manager
        # [FrontendManager] -> Request Command -> RequestManager
        request_backend = context.socket(zmq.PUSH)
        request_backend.connect(relative_channel(RequestManager.FRONTEND_CHANNEL, self.ipc_dir))

        # We are ready for connections
        self.ready.set()

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(response_backend, zmq.POLLIN)

        # Switch messages between sockets
        while True:
            sockets = dict(poller.poll())

            if frontend in sockets:
                request = frontend.recv_multipart()
                request_backend.send_multipart(request)

            if response_backend in sockets:
                response = response_backend.recv_multipart()
                frontend.send_multipart(response)
