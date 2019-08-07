import torch
from torch.multiprocessing.reductions import ForkingPickler

import NetworkManager
import NetworkClient
from time import time

torch.multiprocessing.set_sharing_strategy("file_system")


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
        for _ in range(10):
            h_relu = self.hidden(h_relu).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


output_shape = (3, )
output_type = torch.float32

input_shape = (8, )
input_type = torch.float32

manager = NetworkManager.NetworkManager(TwoLayerNet, input_shape, input_type, output_shape, output_type, placement={'cpu': 4},
                                        network_args=[8, 128, 3], batch_size=32)
manager.start()


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

WORKERS = 64
COUNT = 1000
workers = [TestWorker(manager.client_config, COUNT) for _ in range(WORKERS)]

for worker in workers:
    worker.start()

for worker in workers:
    worker.ready.wait()

t0 = time()
for worker in workers:
    worker.ready.clear()
    worker.start_event.set()

for worker in workers:
    worker.ready.wait()
    worker.join()
t1 = time()
print(f"Remote Time: {t1 - t0}")

# x = torch.rand(1, 8)
# t0 = time()
# for _ in range(COUNT * WORKERS):
#     manager._local_network(x)
# t1 = time()
# print(f"Local Time: {t1 - t0}")


manager.shutdown()