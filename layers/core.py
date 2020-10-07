# -*- coding: utf-8 -*-

import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lambda_func):
        super(Lambda, self).__init__()
        self.lambda_func = lambda_func
    def forward(self, x):
        return self.lambda_func(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
