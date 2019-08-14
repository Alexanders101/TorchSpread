from TorchSpread import NetworkManager, NetworkClient

import torch
from torch import nn
import numpy as np

from typing import Tuple
from multiprocessing import Array, Process
import ctypes

from matplotlib import pyplot as plt


class TwoLayerNet(nn.Module):
    def __init__(self, worker, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.worker = worker
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
        y_pred = self.linear2(h_relu)
        return y_pred


class TestWorker(Process):
    def __init__(self, index, client_config, xbuffer, ybuffer):
        super().__init__()
        self.index = index
        self.client_config = client_config
        self.xbuffer = xbuffer
        self.ybuffer = ybuffer
        self.num_states = xbuffer.shape[0]

    def run(self):
        client = NetworkClient(self.client_config, self.num_states)
        client.register()
        self.ybuffer[:] = client.predict({'x': self.xbuffer}).numpy()

        # for i in range(self.num_states):
        #     self.ybuffer[i] = client.predict({'x': self.xbuffer[i:i + 1]}).numpy()


def shared_array(ctype: type, shape: Tuple[int, ...], lock: bool = False):
    buffer_base = Array(ctype, int(np.prod(shape)), lock=False)
    buffer = np.ctypeslib.as_array(buffer_base).reshape(shape)

    return buffer_base, buffer

NUM_WORKERS = 1024
NUM_STATES = 64

if __name__ == '__main__':
    output_shape = (3,)
    output_type = torch.float32

    input_shape = {'x': (8,)}
    input_type = {'x': torch.float32}

    manager = NetworkManager(input_shape, input_type, output_shape, output_type, 16 * 1024, TwoLayerNet,
                             network_args=[8, 32, 3], placement={'cpu': 1}, num_worker_buffers=4)
    manager.start()

    print("Creating Data.")
    xbase, xbuffer = shared_array(ctypes.c_float, (NUM_WORKERS, NUM_STATES, 8))
    ybase, ybuffer = shared_array(ctypes.c_float, (NUM_WORKERS, NUM_STATES, 3))

    xbuffer[:] = np.random.randn(NUM_WORKERS, NUM_STATES, 8).astype(np.float32)

    print("Running prediction on remote workers.")
    workers = []
    for i in range(NUM_WORKERS):
        workers.append(TestWorker(i, manager.client_config, xbuffer[i], ybuffer[i]))

    for worker in workers:
        worker.start()

    for i, worker in enumerate(workers):
        worker.join()

    print("Running Prediction Locally.")
    x = torch.from_numpy(xbuffer)
    x = torch.reshape(x, (-1, 8))
    with torch.no_grad():
        y = manager.unsynchronized_training_network({'x': x})
    y = torch.reshape(y, (NUM_WORKERS, NUM_STATES, 3)).numpy()
    error = np.abs(ybuffer - y).ravel()

    plt.figure(figsize=(12, 8))
    plt.hist(error, bins=16)
    plt.axvline(error.max(), color='b', linestyle='dashed', label='Maximum Error')
    plt.axvline(np.finfo(np.float32).eps, color='r', linestyle='dashed', label='Machine Epsilon')
    plt.title("Error from local and remote workers.")
    plt.legend()
    plt.show()

    manager.shutdown()
