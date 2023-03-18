from math import prod
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected multi-layer perceptron."""

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

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param)         # weight
            else:
                nn.init.uniform_(param, -0.01, 0.01)    # bias

    def forward(self, x):
        x = x.float().contiguous().to(self.device)
        return self.net(x)

    @property
    def device(self):
        """str: Determine on which device is the model placed upon, CPU or GPU."""
        return next(self.parameters()).device

#