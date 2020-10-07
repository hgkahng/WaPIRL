# -*- coding: utf-8 -*-

import math
import torch.optim as optim
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup & cosine decay.
       Implementation from `pytorch_transformers.optimization.WarmupCosineSchedule`.
       Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
       linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
       Decreases learning rate for 1. to 0. over remaining `t_total - warmup_steps` following a cosine curve.
       If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, min_lr=1e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.min_lr + float(step) / float(max(1.0, self.warmup_steps))
        # Progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return self.min_lr + max(0., 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_optimizer(params, name: str, lr: float, weight_decay: float, **kwargs):
    """Configure optimizer."""
    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, **kwargs):
    """Configure learning rate scheduler."""
    if name == 'cosine':
        warmup_steps = kwargs.get('warmup_steps', 0)
        return WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs)
    else:
        return None
