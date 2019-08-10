import torch
from torch import multiprocessing

from typing import Dict, List, Tuple

import zmq
import pickle
from threading import Thread, Event
from copy import deepcopy

from NetworkSynchronization import SynchronizationWorker
from utilities import make_buffer, load_buffer, unload_buffer, DEBUG, VERBOSE
from utilities import serialize_int, deserialize_int, iterate_window, slice_buffer

mp_ctx = multiprocessing.get_context('forkserver')
Process = mp_ctx.Process

# TESTING Make all multiprocessing stuff threading for debugging purposes
if DEBUG:
    from threading import Thread
    Process = Thread

# TESTING Print all communication
debug_print = print if VERBOSE else (lambda x: x)


class NetworkWorkerRequester(Thread):
    REQUESTER_CHANNEL = "inproc://REQUESTER"
    INTERNAL_CHANNEL = "inproc://REQUESTER_INTERNAL"

    def __init__(self,
                 input_buffers: Dict[str, object],
                 network_inputs: List[object],
                 network_requests: List[List[Tuple]],
                 network_identity: bytes,
                 request_channel: str,
                 context: zmq.Context):
        super(NetworkWorkerRequester, self).__init__()

        # Local storage
        self.input_buffers = input_buffers
        self.network_inputs = network_inputs
        self.num_inputs = len(network_inputs)
        self.network_requests = network_requests
        self.network_identity = network_identity
        self.context = context

        # Receive requests from the request manager and batch them for the network
        # RequestManager -> Batch -> [WorkerRequester]
        self.request_queue: zmq.Socket = context.socket(zmq.DEALER)
        self.request_queue.setsockopt(zmq.IDENTITY, network_identity)
        self.request_queue.connect(request_channel)

        # Inform the network that there is a ready batch in a given input buffer
        # [WorkerRequester] -> Predict Request -> Worker
        self.predict_queue = context.socket(zmq.PUSH)
        self.predict_queue.bind(self.INTERNAL_CHANNEL)

        self.ready = Event()

    def loader_thread(self, input_index: int, ready: Event):
        """ Asynchronous data loading.

        This is especially helpful when we have a small network, large input, and a large number of clients.
        This makes the copying of the predictions into the network buffer parallel.
        """
        # NetworkWorkerRequester -> Request -> [loader_thread]
        request_queue = self.context.socket(zmq.PULL)
        request_queue.connect(self.INTERNAL_CHANNEL)

        # [loader_thread] -> Request -> NetworkWorker
        predict_queue = self.context.socket(zmq.PUSH)
        predict_queue.connect(self.REQUESTER_CHANNEL)

        # Current loader's appropriate input
        network_input = self.network_inputs[input_index]

        # Ready for requests
        ready.set()

        while True:
            # Current Buffers
            network_request = []

            # Wait for new requests
            request = pickle.loads(request_queue.recv())
            debug_print("Requester {}: Loader {} received request.".format(self.network_identity, input_index))

            # Kill Signal for the worker is a single length request
            if len(request) < 2:
                debug_print("Requester {}: Loader {} Shutdown.".format(self.network_identity, input_index))
                break

            # Transfer data to the network's input buffer
            current_size = 0
            for client, size in iterate_window(request, n=2):
                size = deserialize_int(size)

                load_buffer(network_input, self.input_buffers[client], size, current_size)
                network_request.append((client, size, current_size))

                current_size += size

            # Send request to network and move on to next buffer
            self.network_requests[input_index] = network_request
            predict_queue.send_multipart([serialize_int(input_index), serialize_int(current_size)])

        # Pass kill event to network worker. Only the first loader does this.
        if input_index == 0:
            predict_queue.send(serialize_int(0))

    def _shutdown_predictor(self):
        for _ in range(self.num_inputs):
            self.predict_queue.send(pickle.dumps([NetworkWorker.SHUTDOWN]))

    def run(self):
        # Local variables for faster access mid-loop
        request_queue = self.request_queue
        predict_queue = self.predict_queue

        # Start all of the loader threads
        loader_threads = []
        for loader_index in range(self.num_inputs):
            loader_ready = Event()
            loader_thread = Thread(target=self.loader_thread, args=(loader_index, loader_ready))

            loader_thread.start()
            loader_ready.wait()

            loader_threads.append(loader_thread)

        # Ready for commands
        self.ready.set()
        current_loader = 0

        while True:
            request = request_queue.recv()
            debug_print("Requester {} received request.".format(self.network_identity))

            # Kill Switch
            if len(request) <= 1:
                debug_print("Requester {} Shutdown.".format(self.network_identity))
                self._shutdown_predictor()
                break
            else:
                predict_queue.send_multipart([serialize_int(current_loader), request])
                current_loader = (current_loader + 1) % self.num_inputs


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
        self.ready = mp_ctx.Event()

        self.num_inputs = 2

    def run(self) -> None:
        config: Dict = self.config
        device = torch.device(self.device)
        if 'cuda' in self.device:
            torch.cuda.set_device(device)

        # Network Setup
        # ########################################################################################
        # Create buffers
        network_requests = [[] for _ in range(self.num_inputs)]
        network_inputs = [make_buffer(config["batch_size"], config["input_shape"], config["input_type"], device)
                          for _ in range(self.num_inputs)]

        # Initialize local copy of the network
        network = config['network_class'](*config["network_args"], **config["network_kwargs"])
        network = network.to(device)

        # Communication Setup
        # ########################################################################################
        context: zmq.Context = zmq.Context()

        # Get batched request from the requester
        # NetworkWorkerRequester -> Predict Request -> [NetworkWorker]
        request_queue: zmq.Socket = context.socket(zmq.PULL)
        request_queue.bind(NetworkWorkerRequester.REQUESTER_CHANNEL)

        # Inform the Request Manager that this network is ready for more data
        # [NetworkWorker] -> Ready Command -> RequestManager
        ready_queue: zmq.Socket = context.socket(zmq.PUSH)
        ready_queue.connect(config["ready_channel"])

        # Send the finished jobs to the responders
        # [NetworkWorker] -> Finished Batch -> NetworkWorkerResponder
        response_queues = []
        for i in range(self.num_inputs):
            queue = context.socket(zmq.PAIR)
            queue.bind(NetworkWorkerResponder.RESPONDER_CHANNEL + str(i))
            response_queues.append(queue)

        # Worker Setup
        # ########################################################################################
        # Synchronization thread
        synchronization_thread = SynchronizationWorker(network, self.network_index, self.network_identity,
                                                       config['ipc_dir'], context)
        synchronization_thread.start()
        synchronization_thread.ready.wait()
        network_lock = synchronization_thread.network_lock

        # Request Thread
        request_thread = NetworkWorkerRequester(synchronization_thread.input_buffers, network_inputs, network_requests,
                                                self.network_identity, config["request_channel"], context)
        request_thread.start()
        request_thread.ready.wait()

        # Response Threads
        response_threads = []
        for i in range(self.num_inputs):
            thread = NetworkWorkerResponder(i, synchronization_thread.output_buffers, self.device,
                                            config["response_channel"], context)
            thread.start()
            thread.ready.wait()
            response_threads.append(thread)

        # Main Loop
        # ########################################################################################
        # Inform the manager that this network is ready
        self.ready.set()
        for _ in range(self.num_inputs):
            ready_queue.send(self.network_identity, flags=zmq.NOBLOCK)

        current_buffer = 0
        while True:
            current_size = deserialize_int(request_queue.recv())
            debug_print("Network {} received request of size {}".format(self.network_identity, current_size))

            # Kill signal is a zero prediction size
            if current_size < 1:
                debug_print("Network {} Shutdown".format(self.network_identity))
                break

            network_input = network_inputs[current_buffer]
            network_request = deepcopy(network_requests[current_buffer])

            # Predict on the data
            with network_lock:
                with torch.no_grad():
                    network_output = network(slice_buffer(network_input, 0, current_size))
                    # network_output = network(network_input[:current_size])

            # Inform the Request Manager that we are ready for more requests
            ready_queue.send(self.network_identity, flags=zmq.NOBLOCK)

            debug_print("Network {} has finished its prediction.".format(self.network_identity))

            # Send the responses back to the clients
            # response_thread.respond(response_queue, network_request, network_output)
            response_threads[current_buffer].respond(response_queues[current_buffer], network_request, network_output)
            current_buffer = (current_buffer + 1) % self.num_inputs

        # Cleanup
        for thread, queue in zip(response_threads, response_queues):
            thread.shutdown(queue)
            thread.join()
        request_thread.join()
        synchronization_thread.join()


class NetworkWorkerResponder(Thread):
    RESPONDER_CHANNEL = "inproc://RESPONDER"

    def __init__(self, index: int, output_buffers: Dict[str, object], device: str,
                 response_channel: str, context: zmq.Context):
        """ Asynchronous thread for sending out responses to the clients informing them that their prediction
            is finished. This is faster than doing it in the main thread since the network is free while the
            responses are being sent out.

        """
        super(NetworkWorkerResponder, self).__init__()

        # Local Storage
        self.output_buffers = output_buffers
        self.device = device
        self.index = index
        self.gpu = 'cuda' in self.device

        # [NetworkWorkerResponder] -> Done Command -> ResponseManager
        self.response_queue: zmq.Socket = context.socket(zmq.PUSH)
        self.response_queue.connect(response_channel)

        # NetworkWorker -> Done Command -> [NetworkWorkerResponder]
        # [NetworkWorkerResponder] -> Ready Command -> NetworkWorker
        self.request_queue: zmq.Socket = context.socket(zmq.PAIR)
        self.request_queue.connect(self.RESPONDER_CHANNEL + str(index))

        # Variables for storing response
        self.network_request = None
        self.network_output = None

        self.ready = Event()

    def respond(self, response_queue: zmq.Socket, network_request: List[Tuple], network_output: torch.Tensor):
        """ Respond to a new set of clients. """
        response_queue.recv()
        self.network_request = network_request
        self.network_output = network_output
        response_queue.send(b'')

    def shutdown(self, response_queue: zmq.Socket):
        """ Stop this thread. """
        response_queue.recv()
        response_queue.send(b'KILL')

    def run(self):
        self.request_queue.send(b'')
        self.ready.set()

        while True:
            # Wait for an output to be ready
            command = self.request_queue.recv()
            debug_print("Responder {} for {} received response.".format(self.index, self.device))

            # Kill Switch
            if len(command) > 1:
                debug_print("Responder {} for {} Shutdown.".format(self.index, self.device))
                break

            # Copy the results to the output buffers
            # And inform the client that their request is complete
            if self.gpu and False:
                for client, size, index in self.network_request:
                    unload_buffer(self.output_buffers[client], self.network_output, size, index)

                # torch.cuda.synchronize()

                for client, _, _ in self.network_request:
                    self.response_queue.send_multipart([client, b'DONE'])
            else:
                for client, size, index in self.network_request:
                    unload_buffer(self.output_buffers[client], self.network_output, size, index)
                    self.response_queue.send_multipart([client, b'DONE'])

            # Ready for more
            self.request_queue.send(b'')
