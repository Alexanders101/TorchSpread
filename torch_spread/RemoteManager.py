from threading import Thread
from typing import Dict, List, Any

import zmq
import zmq.devices
from torch import multiprocessing

from .NetworkClient import NetworkClient, RemoteCommands
from .utilities import deserialize_int, deserialize_buffer_into, serialize_buffer

mp_ctx = multiprocessing.get_context('forkserver')
Process = mp_ctx.Process


class RemoteHandler(Thread):
    def __init__(self,
                 context: zmq.Context,
                 channel: int,
                 client_config: Dict[str, Any],
                 batch_size: int):
        super(RemoteHandler, self).__init__()

        self.context = context
        self.channel = channel
        self.batch_size = batch_size
        self.client_config = client_config

        self.ready = mp_ctx.Event()

    def run(self) -> None:
        request_queue = self.context.socket(zmq.PULL)
        request_queue.connect(self.interconnect_channel(self.channel))

        response_queue = self.context.socket(zmq.PUSH)
        response_queue.connect(RemoteManager.HANDLER_RETURN)

        def send_error(identity: bytes, message: str):
            response_queue.send_multipart([identity, RemoteCommands.ERROR, message.encode()], copy=False)

        self.ready.set()

        with NetworkClient(self.client_config, self.batch_size) as client:
            while True:
                identity, command, data = request_queue.recv_multipart(copy=False)
                command_byte = command.bytes

                if command_byte == RemoteCommands.REGISTER:
                    response_queue.send_multipart([identity, command, identity], copy=False)

                elif command_byte == RemoteCommands.DEREGISTER:
                    response_queue.send_multipart([identity, command, identity], copy=False)
                    break

                else:
                    # Decompress and store the sent tensor into the local client's input buffer
                    try:
                        size, compress = deserialize_buffer_into(client.input_buffer, data.buffer)
                    except Exception as exception:
                        send_error(identity, f"FAILED_DESERIALIZATION: {exception}")
                        continue

                    # Send the buffer for prediction
                    try:
                        client.predict(size)
                    except Exception as exception:
                        send_error(identity, f"FAILED_PREDICTION: {exception}")
                        continue

                    # Prepare output data to send back to the client
                    data = serialize_buffer(client.output_buffer, compress)

                    response_queue.send_multipart([identity, command, data], copy=False)

    @staticmethod
    def interconnect_channel(channel: int) -> str:
        return f"inproc://handler_{channel}"


class RemoteManager(Process):
    HANDLER_RETURN = "inproc://handler_return"

    def __init__(self, client_config, port: int = 8765, hostname: str = "*"):
        super(RemoteManager, self).__init__()

        self.client_config = client_config
        self.port = port
        self.hostname = hostname

        self.ready = mp_ctx.Event()

        self.handlers: Dict[bytes, RemoteHandler] = {}
        self.queues: Dict[bytes, zmq.Socket] = {}

        self.latest_channel = 0
        self.channels: List[int] = []
        self.channels.append(self.latest_channel)

    def shutdown(self, context):
        queue = context.socket(zmq.REQ)
        queue.connect(f"tcp://localhost:{self.port}")
        queue.send_multipart([RemoteCommands.KILL, b''])

    def _register(self, context: zmq.Context, identity: bytes, data: Any):
        if identity in self.queues:
            return

        # Create a new in-process channel for this handler
        channel = self.channels.pop()
        interconnect = RemoteHandler.interconnect_channel(channel)

        # Create the handler's zmq upstream socket to send data
        queue = context.socket(zmq.PUSH)
        queue.bind(interconnect)

        # Create the handler thread
        handler = RemoteHandler(context, channel, self.client_config, deserialize_int(data))
        handler.start()
        handler.ready.wait()

        # Update the database
        self.handlers[identity] = handler
        self.queues[identity] = queue

        # Put the next available channel to the queue
        # When workers get de-registered they will also add themselves to this queue
        # This lets us reuse channels
        self.latest_channel += 1
        self.channels.append(self.latest_channel)

    def _deregister(self, identity):
        self.handlers[identity].join()
        self.channels.append(self.handlers[identity].channel)
        del self.handlers[identity]
        del self.queues[identity]

    def _send_to_handler(self, identity, command, data):
        self.queues[identity.bytes].send_multipart([identity, command, data], copy=False)

    def _cleanup(self):
        for identity in self.handlers:
            self._deregister(identity)

    def run(self) -> None:
        context: zmq.Context = zmq.Context()

        frontend = context.socket(zmq.ROUTER)
        frontend.bind(f"tcp://{self.hostname}:{self.port}")

        backend = context.socket(zmq.PULL)
        backend.bind(self.HANDLER_RETURN)

        self.ready.set()

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        while True:
            sockets = dict(poller.poll())

            if frontend in sockets:
                identity, _, command, data = frontend.recv_multipart(copy=False)
                command_bytes = command.bytes

                if command_bytes == RemoteCommands.KILL:
                    self._cleanup()
                    break

                # Register worker if we need to
                if identity.bytes not in self.queues:
                    if command_bytes == RemoteCommands.REGISTER:
                        self._register(context, identity.bytes, data.bytes)
                    else:
                        frontend.send_multipart([identity, _, RemoteCommands.ERROR, b"NOREGISTER"], copy=False)
                        continue

                # Send the message to the handler
                self._send_to_handler(identity, command, data)

                # Clean up the handler if it was a de-registration
                if command_bytes == RemoteCommands.DEREGISTER:
                    self._deregister(identity.bytes)

            if backend in sockets:
                identity, command, data = backend.recv_multipart(copy=False)
                frontend.send_multipart([identity, b'', command, data], copy=False)