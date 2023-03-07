from math import prod

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_shape, hidden_sizes, out_size):
        """Init a multi-layer perceptron neural net.

        Args:
            in_shape: list[int]
                Shape of the input. E.g. (3, 32, 32) for images.
            hidden_sizes: list[int]
                List of sizes for the hidden layers of the network.
            out_size: int
                Size of the output.
        """
        super().__init__()
        self.in_size = prod(in_shape)
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size

        # Initialize the model architecture.
        layers = []
        sizes = [self.in_size] + hidden_sizes
        for fan_in ,fan_out in zip(sizes[:-1], sizes[1:]):
            layers.extend([
                nn.Linear(fan_in, fan_out),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(fan_out, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.to(self.device))

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device

#