import torch
from torch import multiprocessing
from torch.multiprocessing import Process

from typing import Dict, List

import zmq
import dill
from threading import Thread, Event

from NetworkSynchronization import SynchronizationWorker
from utilities import make_buffer, load_buffer, unload_buffer
from utilities import serialize_int, deserialize_int


class NetworkWorkerResponder(Thread):
    def __init__(self, output_buffers: Dict[str, object], response_queue: zmq.Socket, device: str):
        """ Asynchronous thread for sending out responses to the clients informing them that their prediction
            is finished. This is faster than doing it in the main thread since the network is free while the
            responses are being sent out.

        Parameters
        ----------
        output_buffers
        response_queue
        """
        super(NetworkWorkerResponder, self).__init__()
        self.output_buffers = output_buffers
        self.output_start = Event()
        self.output_ready = Event()

        self.response_queue = response_queue
        self.device = device
        self.gpu = 'cuda' in self.device

        self.clients = None
        self.sizes = None
        self.indices = None
        self.output_tensor = None

        self.output_ready.set()

        self.ready = Event()
        self.kill_set = False

    def respond(self, clients: List[bytes], sizes: List[int], indices: List[int], output_tensor: torch.Tensor):
        """ Respond to a new set of clients. """
        self.output_ready.wait()
        self.output_ready.clear()

        self.clients = clients
        self.sizes = sizes
        self.indices = indices
        self.output_tensor = output_tensor

        self.output_start.set()

    def shutdown(self):
        """ Stop this thread. """
        self.kill_set = True
        self.output_start.set()

    def run(self):
        if self.gpu:
            stream = torch.cuda.Stream(torch.device(self.device))

        self.ready.set()

        while True:
            # Wait for an output to be ready
            self.output_start.wait()
            self.output_start.clear()

            # Kill Switch
            if self.kill_set:
                break

            # Copy the results to the output buffers
            # And inform the client that their request is complete
            if self.gpu:
                with torch.cuda.stream(stream):
                    for client, size, index in zip(self.clients, self.sizes, self.indices):
                        unload_buffer(self.output_buffers[client], self.output_tensor, size, index)

                    stream.synchronize()

                    for client in self.clients:
                        self.response_queue.send_multipart([client, b'DONE'])
            else:
                for client, size, index in zip(self.clients, self.sizes, self.indices):
                    unload_buffer(self.output_buffers[client], self.output_tensor, size, index)
                    self.response_queue.send_multipart([client, b'DONE'])

            # Ready for more
            self.output_ready.set()


class NetworkWorker(Process):
    # Commands
    READY = b"R"
    SHUTDOWN = b"S"

    def __init__(self, network_index: int, device: str, config: Dict[str, object]):
        super(NetworkWorker, self).__init__()
        self.network_index = network_index
        self.network_identity = b"N" + serialize_int(self.network_index)

        self.config = config
        self.device = device
        self.ready = multiprocessing.Event()

    @staticmethod
    def _iterate_window(iterator: List, n: int = 2):
        size = len(iterator)
        iterator = iter(iterator)
        for _ in range(0, size, n):
            yield (next(iterator) for _ in range(n))

    def run(self) -> None:
        config: Dict = self.config

        # Initialize Communication
        context: zmq.Context = zmq.Context()

        request_queue: zmq.Socket = context.socket(zmq.DEALER)
        request_queue.setsockopt(zmq.IDENTITY, self.network_identity)
        request_queue.connect(config["request_channel"])

        response_queue: zmq.Socket = context.socket(zmq.PUSH)
        response_queue.connect(config["response_channel"])

        ready_queue: zmq.Socket = context.socket(zmq.PUSH)
        ready_queue.connect(config["ready_channel"])

        # Initialize local copy of the network
        network_input = make_buffer(config["batch_size"], config["input_shape"], config["input_type"], self.device)
        network = config['network_class'](*config["network_args"], **config["network_kwargs"])
        network = network.to(self.device)

        # Start synchronization thread
        synchronization_thread = SynchronizationWorker(network, self.network_index, self.network_identity, config['ipc_dir'], context)
        synchronization_thread.start()
        network_lock = synchronization_thread.network_lock
        input_buffers = synchronization_thread.input_buffers

        # Start Response Thread for asynchronous results returns
        response_thread = NetworkWorkerResponder(synchronization_thread.output_buffers, response_queue, self.device)
        response_thread.start()

        # Tell the manager that this network is ready
        ready_queue.send(self.network_identity, flags=zmq.NOBLOCK)
        self.ready.set()

        while True:
            clients = []
            sizes = []
            indices = []

            # Wait for new requests
            request = request_queue.recv_multipart()

            # Kill Signal for the network is a single length request
            if len(request) == 1:
                response_thread.shutdown()
                break

            # Transfer data to the network's input tensor
            current_size = 0
            for client, size in self._iterate_window(request, n=2):
                size = deserialize_int(size)

                # Append data for each client
                clients.append(client)
                sizes.append(size)
                indices.append(current_size)

                # Add this client's input data to the buffer
                load_buffer(network_input, input_buffers[client], size, current_size)
                current_size += size

            # Predict on the data
            with network_lock:
                with torch.no_grad():
                    output_tensor = network(network_input[:current_size])

            # Inform the queue that this worker is ready for more data
            # And start transmitting the output afterwards
            # This give us slightly more latency optimization
            ready_queue.send(self.network_identity, flags=zmq.NOBLOCK)
            response_thread.respond(clients, sizes, indices, output_tensor)

        response_thread.join()
        synchronization_thread.join()
