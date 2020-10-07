# -*- coding: utf-8 -*-

"""
    Base classes.
"""

import torch
import torch.nn as nn


class BackboneBase(nn.Module):
    def __init__(self, layer_config: list, in_channels: int):
        super(BackboneBase, self).__init__()
        assert isinstance(layer_config, (str, list, dict))
        assert in_channels in [1, 2, 3]

    def forward(self, x):
        raise NotImplementedError

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad = False

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])


class HeadBase(nn.Module):
    def __init__(self, output_size: int):
        super(HeadBase, self).__init__()
        assert isinstance(output_size, int)

    def load_weights_from_checkpoint(self, path: str, key: str):
        """
        Load weights from a checkpoint.
        Arguments:
            path: str, path to pretrained `.pt` file.
            key: str, key to retrieve the model from a state dictionary of pretrained modules.
        """
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])
