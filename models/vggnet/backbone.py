# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from models.base import BackboneBase
from utils.initialization import initialize_weights


class VggNetBackbone(BackboneBase):
    def __init__(self, layer_config: list, in_channels: int = 2):
        """
        Arguments:
            layer_config: list, following rules of VGG.
            in_channels: int, number of channels in the input.
            batch_norm: bool, default True.
        """
        super(VggNetBackbone, self).__init__(layer_config, in_channels)

        self.layer_config = layer_config
        self.in_channels = in_channels
        self.layers = self.make_layers(
            layer_cfg=self.layer_config['channels'],
            in_channels=self.in_channels,
            batch_norm=self.layer_config['batch_norm']
        )

        initialize_weights(self.layers, activation='relu')

    def forward(self, x: torch.FloatTensor):
        return self.layers(x)

    @staticmethod
    def make_layers(layer_cfg: list, in_channels: int, batch_norm: bool = True):
        """Expects a list of lists for `cfg`."""

        layers = nn.Sequential()
        for i, block_cfg in enumerate(layer_cfg, 1):
            block = nn.Sequential()
            p_idx = c_idx = 1
            for _, v in enumerate(block_cfg, 1):
                if v == 'M':
                    block.add_module(f'pool{p_idx}', nn.MaxPool2d(2, 2))
                    p_idx += 1
                elif v == 'A':
                    block.add_module(f'pool{p_idx}', nn.AvgPool2d(2, 2))
                    p_idx += 1
                else:
                    assert isinstance(v, int), "Number of filters."
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
                    block.add_module(f'conv{c_idx}', conv2d)
                    if batch_norm:
                        block.add_module(f'bnorm{c_idx}', nn.BatchNorm2d(v))
                    block.add_module(f'relu{c_idx}', nn.ReLU(inplace=True))
                    in_channels = v
                    c_idx += 1
            layers.add_module(f'block{i}', block)

        return layers

    @property
    def out_channels(self):
        for block_cfg in self.layer_config['channels'][::-1]:
            for v in block_cfg[::-1]:
                if isinstance(v, int):
                    return v

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
