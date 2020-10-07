# -*- coding: utf-8 -*-

import os
import json
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

from tasks.base import Task
from utils.loss import WaPIRLLoss
from utils.logging import get_tqdm_config
from utils.logging import make_epoch_description


class MemoryBank(object):
    def __init__(self, size: tuple, device: int or str, weight: float = 0.5):

        self.size = size
        self.device = device
        self.weight = weight

        self.buffer = torch.zeros(*size, device=device)
        self.initialized = False

    def initialize(self,
                   backbone: nn.Module,
                   projector: nn.Module,
                   data_loader: torch.utils.data.DataLoader):
        """Initialize memory bank values with a forward pass of the model."""

        backbone.eval()
        projector.eval()

        with tqdm.tqdm(desc="Initializing memory...", total=len(data_loader), dynamic_ncols=True) as pbar:

            with torch.no_grad():
                for _, batch in enumerate(data_loader):
                    x = batch['x'].to(self.device, non_blocking=True)
                    j = batch['idx']
                    self.buffer[j, :] = projector(backbone(x)).detach()
                    pbar.update(1)

        self.intialized = True  # pylint: disable=attribute-defined-outside-init

    def update(self, index: list, values: torch.FloatTensor):
        """Update memory with exponentially weighted moving average."""
        self.buffer[index, :] = self.weight * self.buffer[index, :] + (1 - self.weight) * values

    def get_representations(self, index: int or tuple or list):
        """Return memory representations corresponding to the provided `index`."""
        return self.buffer[index, :]

    def get_negatives(self, size: int, exclude: int or tuple or list):
        """
        Sample negative examples from memory buffer of size `size`.
        Indices in `exclude` will not be included.
        """
        logits = torch.ones(self.buffer.size(0), device=self.device)
        logits[exclude] = 0
        sample_size = torch.Size([size])
        return self.buffer[Categorical(logits=logits).sample(sample_size), :]

    def save(self, path: str, **kwargs):
        """Save memory represenations to a file."""
        ckpt = {
            'weight': self.weight,
            'buffer': self.buffer.detach().cpu(),
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load(self, path: str):
        """Load memory representations from a file."""
        ckpt = torch.load(path)
        self.weight = ckpt['weight']
        self.buffer = ckpt['buffer'].to(self.device)
        self.initialized = True  # pylint: disable=attribute-defined-outside-init


class WaPIRL(Task):
    def __init__(self,
                 backbone: nn.Module,
                 projector: nn.Module,
                 memory: MemoryBank,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 loss_function: WaPIRLLoss,
                 loss_weight: float = 0.5,
                 num_negatives: int = 5000,
                 distributed: bool = False,
                 local_rank: int = 0,
                 checkpoint_dir: str = None,
                 metrics: dict = None,
                 write_summary: bool = True,
                 **kwargs):
        super(WaPIRL, self).__init__()

        assert isinstance(memory, MemoryBank)
        assert isinstance(loss_function, WaPIRLLoss)

        self.backbone = backbone
        self.projector = projector
        self.memory = memory
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.loss_weight = loss_weight
        self.num_negatives = num_negatives

        self.distributed = distributed
        self.local_rank = local_rank

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.metrics = metrics if isinstance(metrics, dict) else None

        if (self.local_rank == 0) & write_summary:
            self.writer = SummaryWriter(self.checkpoint_dir)
        else:
            self.writer = None

    def run(self,
            train_set: torch.utils.data.Dataset,
            valid_set: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int,
            num_workers: int = 0,
            **kwargs):

        logger = kwargs.get('logger', None)
        save_every = kwargs.get('save_every', epochs)

        self.backbone.to(self.local_rank)
        self.projector.to(self.local_rank)

        if self.distributed:
            raise NotImplementedError
        else:
            train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
            valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)

        # Initialize memory representations for the training data
        if not self.memory.initialized:
            self.memory.initialize(self.backbone, self.projector, train_loader)

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='blue')) as pbar:

            best_valid_loss = float('inf')
            best_epoch = 0

            for epoch in range(1, epochs + 1):

                # 0. Train & evaluate
                train_history = self.train(train_loader)
                valid_history = self.evaluate(valid_loader)

                # 1. Epoch history (loss)
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss')
                    },
                }

                # 2. Epoch history (other metrics if provided)
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, _ in self.metrics.items():
                        epoch_history[metric_name] = {
                            'train': train_history.get(metric_name),
                            'valid': valid_history.get(metric_name),
                        }

                # 3. Tensorboard
                if self.writer is not None:
                    for metric_name, metric_dict in epoch_history.items():
                        self.writer.add_scalars(
                            main_tag=metric_name,
                            tag_scalar_dict=metric_dict,
                            global_step=epoch
                        )
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            tag='lr',
                            scalar_value=self.scheduler.get_last_lr()[0],
                            global_step=epoch
                        )

                # 4-1. Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    if self.local_rank == 0:
                        self.save_checkpoint(self.best_ckpt, epoch=epoch, **epoch_history)
                        self.memory.save(os.path.join(os.path.dirname(self.best_ckpt), 'best_memory.pt'), epoch=epoch)

                # 4-2. Save intermediate models
                if epoch % save_every == 0:
                    if self.local_rank == 0:
                        new_ckpt = os.path.join(self.checkpoint_dir, f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt')
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 5. Update learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # 6. Logging
                desc = make_epoch_description(
                    history=epoch_history,
                    current=epoch,
                    total=epochs,
                    best=best_epoch
                )
                pbar.set_description_str(desc)
                pbar.update(1)

                if logger is not None:
                    logger.info(desc)

        # 7. Save last model
        if self.local_rank == 0:
            self.save_checkpoint(self.last_ckpt, epoch=epoch, **epoch_history)
            self.memory.save(os.path.join(os.path.dirname(self.last_ckpt), 'last_memory.pt'), epoch=epoch)


    def train(self, data_loader: torch.utils.data.DataLoader, **kwargs):  # pylint: disable=unused-argument
        """Train function defined for a single epoch."""

        out = {'loss': 0.}
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=True)

        with tqdm.tqdm(**get_tqdm_config(steps_per_epoch, leave=False, color='green')) as pbar:
            for i, batch in enumerate(data_loader):

                j  = batch['idx']
                x  = batch['x'].to(self.local_rank)
                x_t = batch['x_t'].to(self.local_rank)
                z_concat = self.predict(torch.cat([x, x_t], dim=0))
                z = z_concat[:x.size(0)]
                z_t = z_concat[x.size(0):]

                m = self.memory.get_representations(j).to(self.local_rank)
                negatives = self.memory.get_negatives(self.num_negatives, exclude=j)

                # Calculate loss
                loss_z, _ = self.loss_function(
                    anchors=m,
                    positives=z,
                    negatives=negatives,
                )
                loss_z_t, logits = self.loss_function(
                    anchors=m,
                    positives=z_t,
                    negatives=negatives,
                )
                loss = (1 - self.loss_weight)  * loss_z + self.loss_weight * loss_z_t

                # Backpropagation & update
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.memory.update(j, values=z.detach())

                # Accumulate loss & metrics
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        with torch.no_grad():
                            logits = logits.detach()
                            targets = torch.zeros(logits.size(0), device=logits.device)
                            out[metric_name] += metric_function(logits, targets).item()

                desc = f" Batch - [{i+1:>4}/{steps_per_epoch:>4}]: "
                desc += " | ".join ([f"{k}: {v/(i+1):.4f}" for k, v in out.items()])
                pbar.set_description_str(desc)
                pbar.update(1)

        return {k: v / steps_per_epoch for k, v in out.items()}

    def evaluate(self, data_loader: torch.utils.data.DataLoader, **kwargs):  # pylint: disable=unused-argument
        """Evaluate current model. A single pass through the given dataset."""

        out = {'loss': 0.}
        steps_per_epoch = len(data_loader)
        self._set_learning_phase(train=False)

        with torch.no_grad():
            for _, batch in enumerate(data_loader):

                j   = batch['idx']
                x   = batch['x'].to(self.local_rank)
                x_t = batch['x_t'].to(self.local_rank)
                z_concat = self.predict(torch.cat([x, x_t], dim=0))
                z = z_concat[:x.size(0)]
                z_t = z_concat[x.size(0):]

                negatives = self.memory.get_negatives(self.num_negatives, exclude=j)

                # Note that no memory representation (m) exists for the validation data.
                loss, logits = self.loss_function(
                    anchors=z,
                    positives=z_t,
                    negatives=negatives,
                )

                # Accumulate loss & metrics
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        logits = logits.detach()                                     # (N, 1+ num_negatives)
                        targets = torch.zeros(logits.size(0), device=logits.device)  # (N, )
                        out[metric_name] += metric_function(logits, targets).item()

        return {k: v / steps_per_epoch for k, v in out.items()}

    def predict(self, x: torch.FloatTensor):
        return self.projector(self.backbone(x))

    def _set_learning_phase(self, train: bool = False):
        if train:
            self.backbone.train()
            self.projector.train()
        else:
            self.backbone.eval()
            self.projector.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if \
                self.scheduler is not None else None
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.backbone.load_state_dict(ckpt['backbone'])
        self.projector.load_state_dict(ckpt['projector'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['backbone']
        del ckpt['projector']
        del ckpt['optimizer']
        del ckpt['scheduler']
        return ckpt
