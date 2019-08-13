import torch
from torch import multiprocessing

from typing import Dict, List, Tuple, Union

import zmq
import pickle
from threading import Thread, Event

from .NetworkSynchronization import SynchronizationWorker
from .utilities import make_buffer, load_buffer, unload_buffer, VERBOSE, BufferType
from .utilities import serialize_int, deserialize_int, iterate_window, slice_buffer, send_buffer

mp_ctx = multiprocessing.get_context('forkserver')
Process = mp_ctx.Process

# TESTING Print all communication
debug_print = print if VERBOSE else (lambda x: x)


class RequestWorker(Thread):
    WORKER_REQUEST_CHANNEL = "inproc://REQUEST"

    def __init__(self, index: int, identity: bytes, client_input_buffers: Dict[str, BufferType],
                 batch_input_buffers: List[BufferType], network_input_buffers: List[BufferType],
                 network_requests: List[List[Tuple]], gpu: bool, request_channel: str, context: zmq.Context):
        super(RequestWorker, self).__init__()

        # Local storage
        self.client_input_buffers = client_input_buffers
        self.batch_input_buffer = batch_input_buffers[index]
        self.network_input_buffer = network_input_buffers[index]
        self.network_requests = network_requests
        self.context = context
        self.gpu = gpu

        self.index = index
        self.identity = identity

        # Receive requests from the request manager and copy them to the network buffer
        # RequestManager -> Batch -> [WorkerRequester]
        self.request_queue: zmq.Socket = context.socket(zmq.DEALER)
        self.request_queue.setsockopt(zmq.IDENTITY, self.identity)
        self.request_queue.connect(request_channel)

        # Inform the network that there is a ready batch in a given input buffer
        # [WorkerRequester] -> Predict Request -> Worker
        self.predict_queue = context.socket(zmq.PUSH)
        self.predict_queue.connect(self.WORKER_REQUEST_CHANNEL)

        self.ready = Event()

    def run(self):
        request_index = self.index

        # Ready for commands
        self.ready.set()

        while True:
            # Current Buffers
            network_request = []

            # Wait for new requests
            client_requests = pickle.loads(self.request_queue.recv())
            debug_print("Loader {} received request.".format(self.identity))

            # KILL SIGNAL: length 1 request
            if len(client_requests) < 2:
                debug_print("Loader {} Shutdown.".format(self.identity))
                break

            # Transfer data to the network's input buffer
            current_size = 0
            for client, size in iterate_window(client_requests, n=2):
                size = deserialize_int(size)

                load_buffer(self.batch_input_buffer, self.client_input_buffers[client], size, current_size)
                network_request.append((client, size, current_size))

                current_size += size

            if self.gpu:
                load_buffer(self.network_input_buffer, self.batch_input_buffer, current_size, 0)

            # Send request to network
            self.network_requests[request_index] = network_request
            self.predict_queue.send_multipart([serialize_int(request_index), serialize_int(current_size)])

        # Pass kill event to network worker. Only the first requester does this.
        if request_index == 0:
            self.predict_queue.send_multipart([serialize_int(0), serialize_int(0)])


class NetworkWorker(Process):
    # Commands
    READY = b"R"
    SHUTDOWN = b"S"

    def __init__(self, index: int, device: str, config: Dict[str, Union[int, str]]):
        super(NetworkWorker, self).__init__()
        self.num_buffers = config["num_worker_buffers"]

        self.index = index
        self.identity = b"N" + serialize_int(self.index)
        self.request_worker_identities = [self.identity + serialize_int(i) for i in range(self.num_buffers)]

        self.config = config
        self.device = device
        self.gpu = 'cuda' in device
        self.ready = mp_ctx.Event()

    def _start_synchronization_thread(self, network, context):
        ipc_dir = self.config['ipc_dir']
        synchronization_thread = SynchronizationWorker(network, self.index, self.identity, ipc_dir, context)
        synchronization_thread.start()
        synchronization_thread.ready.wait()
        return synchronization_thread

    def _start_request_threads(self, batch_input_buffers, network_input_buffers, network_requests,
                               client_input_buffers, context):
        request_channel = self.config["request_channel"]
        request_threads = []
        for i in range(self.num_buffers):
            request_thread = RequestWorker(i, self.request_worker_identities[i], client_input_buffers,
                                           batch_input_buffers, network_input_buffers, network_requests,
                                           self.gpu, request_channel, context)
            request_thread.start()
            request_thread.ready.wait()

            request_threads.append(request_thread)
        return request_threads

    def _start_response_threads(self, client_output_buffers, context):
        response_channel = self.config["response_channel"]
        response_threads = []
        for i in range(self.num_buffers):
            thread = ResponseWorker(i, self.device, client_output_buffers, response_channel, context)
            thread.start()
            thread.ready.wait()
            response_threads.append(thread)
        return response_threads

    def run(self) -> None:
        config: Dict = self.config
        device = torch.device(self.device)
        if self.gpu:
            torch.cuda.set_device(device)

        # Network Setup
        # ########################################################################################
        # Create buffers
        network_requests: List[List[Tuple]] = [[] for _ in range(self.num_buffers)]

        # The tensors that are fed into the network
        network_input_buffers = [make_buffer(config["batch_size"], config["input_shape"], config["input_type"], device)
                                 for _ in range(self.num_buffers)]

        # The tensors that the predictions are batched inside of
        if self.gpu:
            batch_input_buffers = [make_buffer(config["batch_size"], config["input_shape"], config["input_type"], 'pin')
                                   for _ in range(self.num_buffers)]
        else:
            batch_input_buffers = network_input_buffers

        # Initialize local copy of the network
        network = config['network_class'](True, *config["network_args"], **config["network_kwargs"])
        network = network.to(device)

        # Communication Setup
        # ########################################################################################
        context: zmq.Context = zmq.Context()

        # Get batched request from the requester
        # NetworkWorkerRequester -> Predict Request -> [NetworkWorker]
        request_queue: zmq.Socket = context.socket(zmq.PULL)
        request_queue.bind(RequestWorker.WORKER_REQUEST_CHANNEL)

        # Inform the Request Manager that this network is ready for more data
        # [NetworkWorker] -> Ready Command -> RequestManager
        ready_queue: zmq.Socket = context.socket(zmq.PUSH)
        ready_queue.connect(config["ready_channel"])

        # Worker Setup
        # ########################################################################################
        synchronization_thread = self._start_synchronization_thread(network, context)
        network_lock = synchronization_thread.network_lock

        request_threads = self._start_request_threads(batch_input_buffers, network_input_buffers, network_requests,
                                                      synchronization_thread.input_buffers, context)

        response_threads = self._start_response_threads(synchronization_thread.output_buffers, context)

        # Main Loop
        # ########################################################################################
        self.ready.set()
        for thread in request_threads:
            ready_queue.send(thread.identity, flags=zmq.NOBLOCK)

        while True:
            request_index, request_size = map(deserialize_int, request_queue.recv_multipart())
            debug_print("Network {} received request of size {}".format(self.identity, request_size))

            # KILL SIGNAL: zero request size
            if request_size < 1:
                debug_print("Network {} Shutdown".format(self.identity))
                break

            # Get the current data
            network_input = network_input_buffers[request_index]
            network_request = network_requests[request_index]

            # Predict on the data
            with network_lock:
                with torch.no_grad():
                    network_output = network(slice_buffer(network_input, 0, request_size))

            # Inform the Request Manager that we are ready for more requests
            ready_queue.send(request_threads[request_index].identity, flags=zmq.NOBLOCK)
            debug_print("Network {} has finished its prediction.".format(self.identity))

            # Send the responses back to the clients
            response_threads[request_index].respond(network_request, network_output)

        # Cleanup after ending
        for thread in response_threads:
            thread.shutdown()
            thread.join()

        for thread in request_threads:
            thread.join()

        synchronization_thread.join()


class ResponseWorker(Thread):
    WORKER_RESPONSE_CHANNEL = "inproc://RESPONSE"

    def __init__(self,
                 index: int,
                 device: str,
                 client_output_buffers: Dict[str, object],
                 response_channel: str,
                 context: zmq.Context):
        """ Asynchronous thread for sending out responses to the clients informing them that their prediction
            is finished. This is faster than doing it in the main thread since the network is free while the
            responses are being sent out.

        """
        super(ResponseWorker, self).__init__()

        # Local Storage
        self.client_output_buffers = client_output_buffers
        self.device = device
        self.index = index
        self.gpu = 'cuda' in self.device

        # [NetworkWorkerResponder] -> Done Command -> ResponseManager
        self.response_queue: zmq.Socket = context.socket(zmq.PUSH)
        self.response_queue.connect(response_channel)

        # [NetworkWorkerResponder] -> Ready Command -> NetworkWorker
        self.request_queue: zmq.Socket = context.socket(zmq.PAIR)
        self.request_queue.bind(self.WORKER_RESPONSE_CHANNEL + str(index))

        # NetworkWorker -> Done Command -> [NetworkWorkerResponder]
        self.worker_queue: zmq.Socket = context.socket(zmq.PAIR)
        self.worker_queue.connect(self.WORKER_RESPONSE_CHANNEL + str(index))

        # Variables for storing response
        self.network_request = None
        self.network_output = None

        self.ready = Event()

    def respond(self, network_request: List[Tuple], network_output: torch.Tensor):
        """ Respond to a new set of clients. """
        self.worker_queue.recv()
        self.network_request = network_request
        self.network_output = network_output
        self.worker_queue.send(b'')

    def shutdown(self):
        """ Stop this thread. """
        self.worker_queue.recv()
        self.worker_queue.send(b'KILL')

    def run(self):
        self.request_queue.send(b'')
        self.ready.set()

        while True:
            # Wait for an output to be ready
            command = self.request_queue.recv()
            debug_print("Responder {} for {} received response.".format(self.index, self.device))

            # KILL SWITCH: Non-empty command
            if len(command) > 1:
                debug_print("Responder {} for {} Shutdown.".format(self.index, self.device))
                break

            # Transfer output to CPU as a batch for efficiency
            self.network_output = send_buffer(self.network_output, 'cpu')

            # Copy the results to the output buffers
            # And inform the client that their request is complete
            for client, size, index in self.network_request:
                unload_buffer(self.client_output_buffers[client], self.network_output, size, index)
                self.response_queue.send_multipart([client, b'DONE'])

            # Ready for more
            self.request_queue.send(b'')
