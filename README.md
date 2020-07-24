# Torch Spread

An minimal python library that implement asynchronous dynamic batching for PyTorch models. 
This is most useful in reinforcement learning, where many small cpu workers interact with an
environment and need to use a model to predict on individual states. Using torch spread, 
these workers can send their states to be batched together and efficiently predicted on multiple
gpu PyTorch models.

## Included Features

- General and simple API that will work with any imaginable PyTorch Module.
- Automatic synchronization of network weights for easy training.
- Efficient IPC communication on a single machine to minimize delay.
- Support for remote network-connected clients with built in compression when sending and receiving tensors.
- Support for both synchronous and asynchronous prediction for clients.
- Minimal API consisting of only three required public classes.

## Requirements

Torch Spread requires Python 3.6 or greater.
Since this library is designed to be as minimal and light as possible, we only have
three dependencies besides Numpy and PyTorch.

- [zeromq](https://pypi.org/project/pyzmq/): Lightweight messaging library.
- [lz4](https://pypi.org/project/lz4/): Fast compression library.
- [msgpack](https://pypi.org/project/msgpack/): Minimal JSON-like serialization library.


## Examples

Examples scripts implement various reinforcement learning algorithms may be found in the `examples/` directory.

## Quickstart

Define a simple network as an example. Note that if you're running this code in a jupyter
notebook, then you need to define the network in a separate `.py` file and import it
for pickling to work.
    
    # example_network.py
    import torch
    from torch import nn
    from torch_spread import SpreadModule
    
    class SmallNetwork(SpreadModule):
        def __init__(self, worker: bool):
            super(SmallNetwork, self).__init__(worker)
            self.linear = nn.Linear(16, 1)
            
        def forward(self, input_buffer):
            x, y = input_buffer['x'], input_buffer['y']
            return {"output": self.linear(x), "sum": x + y}
            
Now define the buffer shapes and dtypes, as well as the batch size for the worker networks.
Here we demonstrate how dictionary buffers can look like, although this would also easily
work with a single tensor input.
We also define where to place the worker networks: in this case, we will create a single
worker network and place it on our gpu. If you have more than one gpu, increase
the number of networks in the round-robin placement.

    import torch
    from torch import nn
    from torch_spread import NetworkClient, NetworkManager, PlacementStrategy
    from example_network import SmallNetwork
    
    input_shape = {'x': (16,), 'y': (16,)}
    input_type = {'x': torch.float32, 'y': torch.float32}
    
    output_shape = {'output': (1,), 'sum': (16,)}
    output_type = {'output': torch.float32, 'sum': torch.float32}
    
    batch_size = 8
    
    manager_arguments = [input_shape, input_type, output_shape, output_type, batch_size, SmallNetwork]
    
    if torch.cuda.is_available():
        placement = PlacementStrategy.round_robin_gpu_placement(1)
    else:
        placement = PlacementStrategy.uniform_cpu_placement(1)
    
Finally, create the network manager, client, and perform a prediction.

    with NetworkManager(*manager_arguments, placement=placement) as manager:
        with NetworkClient(manager.client_config, batch_size) as client:
            data = {'x': torch.rand(batch_size, 16), 'y': torch.rand(batch_size, 16)}
            output = client.predict(data)
            
            print("\n=== INPUT " + "=" * 40)
            print(data["x"])
            
            print("\n=== OUTPUT " + "=" * 40)
            print(output["output"])
            
The only information that a client needs to be created is `manager.client_config`, a small
dictionary containing information on how to communicate with the manager. This dictionary
can be sent to Processes or Threads to create parallel workers (See the Q-learning example).

## Tensor Buffer Format

In order to support dynamic batching across many different workers, your pytorch module must
accept and return a fixed shape tensor buffer. This tensor buffer may be recursively defined as:

    tensor_buffer = Union[torch.Tensor, List[tensor_buffer], Dict[str, tensor_buffer]]

Essentially: any combination of PyTorch tensors, lists, and dictionaries. 
When creating a torch spread manager, you will need to provide an identically structured
buffer defining the shape and dtype of your input and output tensors. Pay special attention
to the shape definition: since a shape is simply a tuple of dimension sizes, the shape buffer
may never use tuples as a storage element. It must only use lists and dictionaries. Also note
that **the shape does not include the batch dimension**. 
    
    shape_buffer = Union[Tuple[int, ...], List[shape_buffer], Dict[str, shape_buffer]]
    dtype_buffer = Union[torch.dtype, List[dtype_buffer], Dict[str, dtype_buffer]]
    
These must be determined ahead of time in order to create your network manager. Hopefully, 
the defined tensor buffer format is flexible enough to allow essentially any type of input
or output from a neural network.

Additionally, we provide a helper class, torch_spread.Buffer, for managing buffers with a simple api.

## PyTorch Module Format

Torch spread will work with most PyTorch modules to remain generalizable, however, we require
two peculiarities:

1. The `__init__` method for your module must accept `worker: bool` as its first argument.
   This input will be `True` if the module instance is a prediction worker and `False` if it
   is the primary synchronization / training network.
   
2. The `forward` method for your module must accept a single buffer as an input and return a
   single buffer as its output. These buffers must batch the shape and dtype definitions you
   provide to the manager.
   
We provide a small helper class `torch_spread.SpreadModule` which you can subclass to 
enforce these two requirements. Its use is optional but helpful to ensure compatibility.


