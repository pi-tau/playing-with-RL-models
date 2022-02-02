import torch
import torch.nn as nn
from math import ceil
from .base_network import BaseNetwork


def _conv2d_outshape(inshape, out_channels, kernel_size, stride):
    nC, nH, nW = inshape
    outH = ((nH - (kernel_size - 1)) / stride)
    outW = ((nW - (kernel_size - 1)) / stride)
    return (out_channels, ceil(outH), ceil(outW))


class ConvolutionalNet(nn.Module, BaseNetwork):
    """ Follows the architecture in DeepMind's paper 'Human-level Control Trough
    Deep Reinforcement Learning'. """

    def __init__(self, input_shape, output_shape):
        super().__init__()
        # OpenGym inputs are (4, 210, 160)
        if len(input_shape) == 2:
            input_shape = (1,) + input_shape
        self.input_shape = input_shape
        self.kwargs = dict(input_shape=input_shape,
                           output_shape=output_shape)
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=64,
            kernel_size=8,
            stride=4,
            padding=0
        )
        conv_shape = _conv2d_outshape(input_shape, 64, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )
        conv_shape = _conv2d_outshape(conv_shape, 64, 4, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        conv_shape = _conv2d_outshape(conv_shape, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=conv_shape[0] * conv_shape[1] * conv_shape[2],
            out_features=1024)
        self.relu4 = nn.ReLU()
        self.dense2 = nn.Linear(
            in_features=1024,
            out_features=512
        )
        self.relu5 = nn.ReLU()
        self.output_layer = nn.Linear(
            in_features=512,
            out_features=output_shape
        )

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.dense1(x))
        x = self.relu5(self.dense2(x))
        out = self.output_layer(x)
        return out
