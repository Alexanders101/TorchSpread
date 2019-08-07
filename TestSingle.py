import NetworkManager

import zmq
import torch
from time import time

import NetworkClient

from NetworkManager import RequestManager, ResponseManager


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.hidden = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        for _ in range(2):
            h_relu = self.hidden(h_relu).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == '__main__':
    output_shape = (3,)
    output_type = torch.float32

    input_shape = (8,)
    input_type = torch.float32

    manager = NetworkManager.NetworkManager(TwoLayerNet, input_shape, input_type, output_shape, output_type, D_in=8,
                                            H=32, D_out=3, batch_size=1024)
    manager.start()

    from utilities import serialize_tensor, serialize_int

    client = NetworkClient.NetworkClient(manager, 1024)

    client.synchronization_response.send_multipart(
        [b'R', serialize_tensor([client.input_buffer, client.output_buffer])])
    network, identity = client.synchronization_response.recv_multipart()

    client.response_queue.setsockopt(zmq.IDENTITY, identity)
    client.synchronization_response.setsockopt(zmq.IDENTITY, identity)
    client.request_queue.connect(RequestManager.FRONTEND_CHANNEL)
    client.response_queue.connect(ResponseManager.FRONTEND_CHANNEL)

    N = 1_000
    t0 = time()
    for i in range(N):
        client.request_queue.send_multipart([identity, serialize_int(1024)])
        client.response_queue.recv()
    t1 = time()
    print(f"Average Time Remote: {1000 * (t1 - t0) / N} ms")

    manager.shutdown()

    # local_network = TwoLayerNet(8, 32, 3)
    # x = torch.rand(1024, 8)
    #
    # t0 = time()
    # for i in range(N):
    #     local_network(x)
    # t1 = time()
    # print(f"Average Time Local: {1000 * (t1 - t0) / N} ms")
