import torch
from torch import nn, multiprocessing
from torch.nn import functional as F
from torch.multiprocessing.reductions import ForkingPickler

import NetworkManager
import NetworkClient
from time import time, sleep


BATCH_SIZE = 16
WORKERS = 64
COUNT = 100


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
        h_relu = self.linear1(x['x']).clamp(min=0)
        for _ in range(10):
            h_relu = self.hidden(h_relu).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 7)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 3)

        self.output1 = nn.Linear(4 * 4 * 256, 256)
        self.output2 = nn.Linear(256, 256)
        self.output3 = nn.Linear(256, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        y = self.conv3(y)
        y = F.relu(y)

        y = y.view(y.size(0), -1)
        y = self.output1(y)

        for _ in range(10):
            y = self.output2(y)

        return self.output3(y)


class TestWorker(torch.multiprocessing.Process):
    def __init__(self, config, count=1_000):
        super().__init__()
        self.config = config
        self.count = count

        self.ready = torch.multiprocessing.Event()
        self.start_event = torch.multiprocessing.Event()

    def run(self):
        client = NetworkClient.NetworkClient(self.config, 1)
        client.register()

        self.ready.set()

        self.start_event.wait()
        self.start_event.clear()

        for _ in range(self.count):
            client.predict_inplace()

        self.ready.set()


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    output_shape = (1, )
    output_type = torch.float32

    input_shape = (3, 16, 16)
    input_type = torch.float32

    try:
        placement = NetworkManager.PlacementStrategy.round_robin_gpu_placement(num_networks=4)
    except ValueError:
        placement = {'cpu': 4}

    manager = NetworkManager.NetworkManager(input_shape, input_type, output_shape, output_type, BATCH_SIZE,
                                            ConvNet, placement=placement)
    with manager:
        workers = [TestWorker(manager.client_config, COUNT) for _ in range(WORKERS)]

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.ready.wait()

        sleep(3)

        t0 = time()
        for worker in workers:
            worker.ready.clear()
            worker.start_event.set()

        for worker in workers:
            worker.ready.wait()
            worker.join()
        t1 = time()
        print(f"Remote Time: {t1 - t0}")

        x = torch.rand(1, *input_shape)
        t0 = time()
        for _ in range(COUNT * WORKERS):
            manager._local_network(x.to(device))
        t1 = time()
        print(f"Local Time: {t1 - t0}")