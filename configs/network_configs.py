# -*- coding: utf-8 -*-

"""
    Configurations for CNN architectures.
"""


ALEXNET_BACKBONE_CONFIGS = ALEXNET_ENCODER_CONFIGS = dict(
    batch_norm='bn',
    local_response_norm='lrn',
)


VGGNET_BACKBONE_CONFIGS = VGGNET_ENCODER_CONFIGS = {
    '16': {
        'channels': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
        'batch_norm': False,
    },
    '16.bn': {
        'channels': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
        'batch_norm': True,
    }
}


RESNET_BACKBONE_CONFIGS = RESNET_ENCODER_CONFIGS = {
    '18': {
        'block_type': 'basic',
        'channels': [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2,
        'strides': [1, 1] + [2, 1] + [2, 1] + [2, 1]
    },
    '50': {
        'block_type': 'bottleneck',
        'channels': [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3,
        'strides': [1, 1, 1] + [2, 1, 1, 1] + [2, 1, 1, 1, 1, 1] + [2, 1, 1]
    },
}
