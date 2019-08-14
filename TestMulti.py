import torch
from torch import nn, multiprocessing
from torch.nn import functional as F

from TorchSpread import NetworkClient, NetworkManager, PlacementStrategy
from time import time, sleep

from multiprocessing import Value
import ctypes


BATCH_SIZE = 8
WORKERS = 64
COUNT = 100


class ConvNet(torch.nn.Module):
    def __init__(self, worker):
        super(ConvNet, self).__init__()

        self.worker = worker

        self.conv1 = nn.Conv2d(3, 32, 7, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2)

        self.output1 = nn.Linear(1024, 256)
        self.output2 = nn.Linear(256, 256)
        self.output3 = nn.Linear(256, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        y = self.conv3(y)
        y = F.relu(y)

        y = self.conv4(y)
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
        self.time = Value(ctypes.c_double, lock=False)

    def run(self):
        with NetworkClient(self.config, 1) as client:
            self.ready.set()

            self.start_event.wait()
            self.start_event.clear()

            t0 = time()
            for _ in range(self.count):
                client.predict_inplace()
            t1 = time()
            self.time.value = t1 - t0

        self.ready.set()


def main(batch_size: int, num_workers: int, repeat_count: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    output_shape = (1, )
    output_type = torch.float32

    input_shape = (3, 64, 64)
    input_type = torch.float32

    if torch.cuda.is_available():
        placement = PlacementStrategy.round_robin_gpu_placement(num_networks=1)
    else:
        placement = {'cpu': 1}

    manager = NetworkManager(input_shape, input_type, output_shape, output_type, batch_size,
                             ConvNet, placement=placement, num_worker_buffers=2)
    with manager:
        workers = [TestWorker(manager.client_config, repeat_count) for _ in range(num_workers)]

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.ready.wait()

        sleep(3)

        average_time = 0

        for worker in workers:
            worker.ready.clear()
            worker.start_event.set()

        for worker in workers:
            worker.ready.wait()
            average_time += worker.time.value
            worker.join()

        print(f"Remote Time: {average_time / num_workers}")

        t0 = time()
        with torch.no_grad():
            for _ in range(repeat_count * num_workers):
                x = torch.rand(1, *input_shape)
                y = manager.unsynchronized_training_network(x.to(device))
                y = y.cpu()
        t1 = time()
        print(f"Equivalent Local Time: {t1 - t0}")

        t0 = time()
        with torch.no_grad():
            for _ in range(repeat_count * num_workers // batch_size):
                x = torch.rand(batch_size, *input_shape)
                y = manager.unsynchronized_training_network(x.to(device))
                y = y.cpu()
        t1 = time()
        print(f"Batch Local Time: {t1 - t0}")


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    main(BATCH_SIZE, WORKERS, COUNT)
