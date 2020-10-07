# -*- coding: utf-8 -*-

import collections
import torch.nn as nn

from models.base import BackboneBase
from utils.initialization import initialize_weights


class AlexNetBackbone(BackboneBase):
    def __init__(self, layer_config: str, in_channels: int = 2):
        super(AlexNetBackbone, self).__init__(layer_config, in_channels)
        self.in_channels = in_channels
        self.norm_type = layer_config
        self.layers = self.make_layers(
            norm_type=self.norm_type,
            in_channels=in_channels
        )

        initialize_weights(self.layers, activation='relu')

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def make_layers(cls, norm_type: str, in_channels: int):

        if norm_type not in ['bn', 'lrn']:
            raise ValueError

        def get_normalization_layer(name: str, num_features: int):
            if name == 'bn':
                return nn.BatchNorm2d(num_features)
            elif name == 'lrn':
                raise NotImplementedError
            else:
                raise NotImplementedError

        bias = norm_type != 'bn'

        block1 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2, bias=bias)),
                    ('norm', get_normalization_layer(norm_type, num_features=96)),
                    ('relu', nn.ReLU()),
                    ('pool', nn.MaxPool2d(kernel_size=3, stride=2))
                ]
            )
        )

        block2 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=bias)),
                    ('norm', get_normalization_layer(norm_type, num_features=256)),
                    ('relu', nn.ReLU()),
                    ('pool', nn.MaxPool2d(kernel_size=3, stride=2))
                ]
            )
        )

        block3 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=bias)),
                    ('norm', get_normalization_layer(norm_type, num_features=384)),
                    ('relu', nn.ReLU()),
                ]
            )
        )

        block4 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=bias)),
                    ('norm', get_normalization_layer(norm_type, num_features=384)),
                    ('relu', nn.ReLU()),
                ]
            )
        )

        block5 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False)),
                    ('norm', get_normalization_layer(norm_type, num_features=256)),
                    ('relu', nn.ReLU()),
                    ('pool', nn.MaxPool2d(kernel_size=3, stride=2))
                ]
            )
        )

        return nn.Sequential(*[block1, block2, block3, block4, block5])

    @property
    def out_channels(self):
        return 256

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
