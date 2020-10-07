# -*- coding: utf-8 -*-

import os
import sys

import torch

from datasets.wm811k import WM811KForWaPIRL
from datasets.transforms import WM811KTransform

from configs.task_configs import WaPIRLConfig
from configs.network_configs import ALEXNET_BACKBONE_CONFIGS
from configs.network_configs import VGGNET_BACKBONE_CONFIGS
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import LinearHead, MLPHead
from tasks.wapirl import WaPIRL, MemoryBank
from utils.loss import WaPIRLLoss
from utils.metrics import TopKAccuracy
from utils.logging import get_logger
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
}

PROJECTOR_TYPES = {
    'linear': LinearHead,
    'mlp': MLPHead,
}


def main():
    """Main function."""

    # 1. Configurations
    config = WaPIRLConfig.parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config.gpus])
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)
    config.save()

    if config.distributed:
        raise NotImplementedError
    else:
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: object):

    torch.cuda.set_device(local_rank)
    if config.distributed:
        raise NotImplementedError

    config.batch_size = config.batch_size // config.num_gpus_per_node
    config.num_workers = max(1, config.num_workers // config.num_gpus_per_node)

    in_channels = int(config.decouple_input) + 1

    # Model
    BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[config.backbone_type]
    Projector = PROJECTOR_TYPES[config.projector_type]
    encoder = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels=in_channels)
    head = Projector(encoder.out_channels, config.projector_size)

    # Optimization
    params = [{'params': encoder.parameters()}, {'params': head.parameters()}]
    optimizer = get_optimizer(
        params=params,
        name=config.optimizer,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        name=config.scheduler,
        epochs=config.epochs,
        warmup_steps=config.warmup_steps
    )

    # Data
    data_kwargs = {
        'transform': WM811KTransform(size=config.input_size, mode='test'),
        'positive_transform': WM811KTransform(size=config.input_size, mode=config.augmentation),
        'decouple_input': config.decouple_input,
    }
    train_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/unlabeled/train/', **data_kwargs),
        WM811KForWaPIRL('./data/wm811k/labeled/train/', **data_kwargs),
    ])
    valid_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/unlabeled/valid/', **data_kwargs),
        WM811KForWaPIRL('./data/wm811k/labeled/valid/', **data_kwargs),
    ])
    test_set = torch.utils.data.ConcatDataset([
        WM811KForWaPIRL('./data/wm811k/unlabeled/test/', **data_kwargs),
        WM811KForWaPIRL('./data/wm811k/labeled/test/', **data_kwargs),
    ])

    # Experiment (WaPIRL)
    experiment_kwargs = {
        'backbone': encoder,
        'projector': head,
        'memory': MemoryBank(
            size=(len(train_set), config.projector_size),
            device=local_rank
            ),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': WaPIRLLoss(temperature=config.temperature),
        'loss_weight': config.loss_weight,
        'num_negatives': config.num_negatives,
        'distributed': config.distributed,
        'local_rank': local_rank,
        'metrics': {
            'top@1': TopKAccuracy(num_classes=1 + config.num_negatives, k=1),
            'top@5': TopKAccuracy(num_classes=1 + config.num_negatives, k=5)
            },
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
    }
    experiment = WaPIRL(**experiment_kwargs)

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_logger(stream=False, logfile=logfile)
        logger.info(f"Data: {config.data}")
        logger.info(f"Augmentation: {config.augmentation}")
        logger.info(f"Observations: {len(train_set):,}")
        logger.info(f"Trainable parameters ({encoder.__class__.__name__}): {encoder.num_parameters:,}")
        logger.info(f"Trainable parameters ({head.__class__.__name__}): {head.num_parameters:,}")
        logger.info(f"Projection head: {config.projector_type} ({config.projector_size})")
        logger.info(f"Checkpoint directory: {config.checkpoint_dir}")
    else:
        logger = None

    # Train (WaPIRL)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'logger': logger,
        'save_every': config.save_every,
    }
    experiment.run(**run_kwargs)

    if logger is not None:
        logger.handlers.clear()


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
