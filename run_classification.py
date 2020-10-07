# -*- coding: utf-8 -*-

import os
import sys

import torch
import numpy as np

from datasets.wm811k import WM811K
from datasets.transforms import WM811KTransform

from configs.task_configs import ClassificationConfig
from configs.network_configs import ALEXNET_BACKBONE_CONFIGS
from configs.network_configs import VGGNET_BACKBONE_CONFIGS
from configs.network_configs import RESNET_BACKBONE_CONFIGS
from models.alexnet import AlexNetBackbone
from models.vggnet import VggNetBackbone
from models.resnet import ResNetBackbone
from models.head import LinearClassifier

from tasks.classification import Classification

from utils.loss import LabelSmoothingLoss
from utils.logging import get_logger
from utils.metrics import MultiAccuracy, MultiF1Score
from utils.optimization import get_optimizer, get_scheduler


AVAILABLE_MODELS = {
    'alexnet': (ALEXNET_BACKBONE_CONFIGS, AlexNetBackbone),
    'vggnet': (VGGNET_BACKBONE_CONFIGS, VggNetBackbone),
    'resnet': (RESNET_BACKBONE_CONFIGS, ResNetBackbone),
}


def main():
    """Main function."""

    # Configurations
    config = ClassificationConfig.parse_arguments()
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

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_logger(stream=False, logfile=logfile)
    else:
        logger = None

    in_channels = int(config.decouple_input) + 1
    num_classes = 9

    # 2. Dataset
    train_transform = WM811KTransform(size=config.input_size, mode=config.augmentation)
    test_transform  = WM811KTransform(size=config.input_size, mode='test')
    train_set = WM811K('./data/wm811k/labeled/train/',
                       transform=train_transform,
                       proportion=config.label_proportion,
                       decouple_input=config.decouple_input)
    valid_set = WM811K('./data/wm811k/labeled/valid/',
                       transform=test_transform,
                       decouple_input=config.decouple_input)
    test_set  = WM811K('./data/wm811k/labeled/test/',
                       transform=test_transform,
                       decouple_input=config.decouple_input)

    # 3. Model
    BACKBONE_CONFIGS, Backbone = AVAILABLE_MODELS[config.backbone_type]
    backbone = Backbone(BACKBONE_CONFIGS[config.backbone_config], in_channels=in_channels)
    classifier = LinearClassifier(in_channels=backbone.out_channels, num_classes=num_classes)

    # 3-1. Load pre-trained weights (if provided)
    if config.pretrained_model_file is not None:
        try:
            backbone.load_weights_from_checkpoint(path=config.pretrained_model_file, key='backbone')
        except KeyError:
            backbone.load_weights_from_checkpoint(path=config.pretrained_model_file, key='encoder')
        finally:
            if logger is not None:
                logger.info(f"Loaded pre-trained model from: {config.pretrained_model_file}")
    else:
        if logger is not None:
            logger.info("No pre-trained model provided.")

    # 3-2. Finetune or freeze weights of backbone
    if config.freeze:
        backbone.freeze_weights()
        if logger is not None:
            logger.info("Freezing backbone weights.")


    # 4. Optimization
    params = [{'params': backbone.parameters()}, {'params': classifier.parameters()}]
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

    # 5. Experiment (classification)
    experiment_kwargs = {
        'backbone': backbone,
        'classifier': classifier,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_function': LabelSmoothingLoss(num_classes, smoothing=config.label_smoothing),
        'distributed': config.distributed,
        'local_rank': local_rank,
        'checkpoint_dir': config.checkpoint_dir,
        'write_summary': config.write_summary,
        'metrics': {
            'accuracy': MultiAccuracy(num_classes=num_classes),
            'f1': MultiF1Score(num_classes=num_classes, average='macro'),
        },
    }
    experiment = Classification(**experiment_kwargs)

    # 6. Run (classification)
    run_kwargs = {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'logger': logger,
    }
    experiment.run(**run_kwargs)
    logger.handlers.clear()


if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
