import itertools

import torch
from torch import nn

from abc import ABC, abstractmethod
from typing import Union, Dict
from collections import OrderedDict


class DistributedModule(nn.Module, ABC):
    """ Helper class to ensure you remember to put worker as the first parameter. """
    def __init__(self, worker: bool):
        super(DistributedModule, self).__init__()
        self.worker = worker


class PlacementStrategy:
    """
    Namespace containing various automated placement strategies for
    placing networks onto many-gpu or many-cpu machines.
    """
    @staticmethod
    def _get_available_gpus():
        num_gpu = torch.cuda.device_count()
        if num_gpu < 1:
            raise ValueError("This machine has no GPUs installed. Cannot use gpu placement strategy.")
        return ["cuda:{}".format(i) for i in range(num_gpu)]

    @staticmethod
    def round_robin_gpu_placement(num_networks: int = 1):
        """ Place networks sequentially across all available GPUs.

        Parameters
        ----------
        num_networks: int
            Total number of networks to place.
        """
        gpus = PlacementStrategy._get_available_gpus()
        placement = {gpu: 0 for gpu in gpus}

        for _, gpu in zip(range(num_networks), itertools.cycle(gpus)):
            placement[gpu] += 1

        return placement

    @staticmethod
    def uniform_gpu_placement(networks_per_gpu=1):
        """ Place an equal number of networks on all available GPU.
        Parameters
        ----------
        networks_per_gpu: int
            Number of networks to place on each GPU.
            Total number of networks will be #GPU * networks_per_gpu
        """
        return {gpu: networks_per_gpu for gpu in PlacementStrategy._get_available_gpus()}

    @staticmethod
    def uniform_cpu_placement(num_networks: int = 1):
        """ Place a number of networks on only cpus.

        Parameters
        ----------
        num_networks: int
            Total number of networks to place.
        """
        return {'cpu': num_networks}


class TrainingWrapper:
    def wrap_network(self, network: nn.Module) -> nn.Module:
        return network

    def wrap_state_dict(self, state_dict: Union[Dict, OrderedDict]) -> Union[Dict, OrderedDict]:
        return state_dict


class DataParallelWrapper(TrainingWrapper):
    def __init__(self, device_ids=None, output_device=None, dim=0):
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim

    def wrap_network(self, network):
        return nn.DataParallel(network, self.device_ids, self.output_device, self.dim)

    def wrap_state_dict(self, state_dict):
        return {'.'.join(key.split('.')[1:]): value for key, value in state_dict.items()}

