# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from datasets.samplers import ImbalancedDatasetSampler


def balanced_loader(dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    drop_last: bool = False,
                    pin_memory: bool = False
                    ):
    """Returns a `DataLoader` instance, which yields a class-balanced minibatch of samples."""

    sampler = ImbalancedDatasetSampler(dataset)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      sampler=sampler,
                      num_workers=num_workers,
                      drop_last=drop_last,
                      pin_memory=pin_memory)
