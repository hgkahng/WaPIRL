# -*- coding: utf-8 -*-

import collections
import torch.nn as nn

from models.base import BackboneBase
from utils.initialization import initialize_weights


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    """3x3 conv. padding is fixed to 1."""
    conv_kws = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    return nn.Conv2d(**conv_kws)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    """1x1 conv. padding is fixed to 1."""
    conv_kws = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False
    )
    return nn.Conv2d(**conv_kws)


def downsample1x1(in_channels: int, out_channels: int, stride: int = 2):
    """1x1 conv + batch normalization. Used for identity mappings. No padding."""
    return nn.Sequential(
        conv1x1(in_channels, out_channels, stride),
        nn.BatchNorm2d(out_channels)
    )


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, **kwargs):
        super(BasicBlock, self).__init__()

        width = kwargs.get('width', 1)

        self.conv1 = conv3x3(in_channels, out_channels * width, stride=stride)  # if stride=2, spatial resolution /= 2.
        self.bnorm1 = nn.BatchNorm2d(out_channels * width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels * width, out_channels, stride=1)      # stride=1, fixed always.
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.stride = stride
        if self.stride != 1:
            self.downsample = downsample1x1(in_channels, out_channels, stride)
        else:
            self.downsample = None

    def forward(self, x):

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.bnorm1(self.conv1(x))
        out = self.relu1(out)
        out = self.bnorm2(self.conv2(out))

        return self.relu2(out + identity)


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(BottleNeck, self).__init__()

        width = kwargs.get('width', 1)

        self.conv1 = conv1x1(in_channels, out_channels * width, stride=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels * width)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels * width, out_channels * width, stride)
        self.bnorm2 = nn.BatchNorm2d(out_channels * width)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(out_channels * width, out_channels * self.expansion, stride=1)
        self.bnorm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride
        if (self.stride != 1) or (in_channels != out_channels * self.expansion):
            self.downsample = downsample1x1(in_channels, out_channels * self.expansion, stride)
        else:
            self.downsample = None

    def forward(self, x):

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.bnorm1(self.conv1(x))
        out = self.relu1(out)
        out = self.bnorm2(self.conv2(out))
        out = self.relu2(out)
        out = self.bnorm3(self.conv3(out))

        return self.relu3(out + identity)


class ResNetBackbone(BackboneBase):
    blocks = {
        'basic': BasicBlock,
        'bottleneck': BottleNeck,
    }
    def __init__(self, layer_config: dict, in_channels: int = 2):
        super(ResNetBackbone, self).__init__(layer_config, in_channels)
        self.in_channels = in_channels
        self.layer_config = layer_config
        self.first_conv = layer_config.get('first_conv', 7)
        self.width = layer_config.get('width', 1)
        self.layers = self.make_layers(
            layer_cfg=self.layer_config,
            in_channels=self.in_channels,
            width=self.width,
            first_conv=self.first_conv,
        )

        initialize_weights(self.layers, activation='relu')

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def make_layers(cls, layer_cfg: dict, in_channels: int, **kwargs):

        out_channels = 64
        layers = nn.Sequential()

        if kwargs['first_conv'] == 3:
            block0 = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                        ('bnorm1', nn.BatchNorm2d(out_channels)),
                        ('relu1', nn.ReLU(inplace=True)),
                    ]
                )
            )
        else:
            block0 = nn.Sequential(
                collections.OrderedDict(
                    [
                        ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                        ('bnorm1', nn.BatchNorm2d(out_channels)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                    ]
                )
            )

        layers.add_module('block0', block0)
        in_channels = out_channels  # 64

        Block = cls.blocks[layer_cfg['block_type']]
        for i, v in enumerate(layer_cfg['channels']):
            stride = layer_cfg['strides'][i]
            layers.add_module(f'block{i+1}', Block(in_channels, v, stride=stride, **kwargs))
            in_channels = v * Block.expansion

        return layers

    @property
    def expansion(self):
        return self.blocks[self.layer_config['block_type']].expansion

    @property
    def out_channels(self):
        return self.layer_config['channels'][-1] * self.expansion

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_output_size(self, input_size: tuple):
        h, w = input_size
        if self.first_conv == 7:
            return h // 32, w // 32
        elif self.first_conv == 3:
            return h // 8, w // 8
