# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import albumentations as A

from torch.distributions import Bernoulli
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform


class ToWBM(BasicTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):  # pylint: disable=unused-argument
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if isinstance(img, torch.ByteTensor):
                img = img.float().div(255)
        return torch.ceil(img * 2)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class MaskedBernoulliNoise(ImageOnlyTransform):
    def __init__(self, noise: float, always_apply: bool = False, p: float = 1.0):
        super(MaskedBernoulliNoise, self).__init__(always_apply, p)
        self.noise = noise
        self.min_ = 0
        self.max_ = 1
        self.bernoulli = Bernoulli(probs=noise)

    def apply(self, x: torch.Tensor, **kwargs):  # pylint: disable=unused-argument
        assert x.ndim == 3
        m = self.bernoulli.sample(x.size()).to(x.device)
        m = m * x.gt(0).float()
        noise_value = 1 + torch.randint_like(x, self.min_, self.max_ + 1).to(x.device)  # 1 or 2
        return x * (1 - m) + noise_value * m

    def get_params(self):
        return {'noise': self.noise}


class WM811KTransform(object):
    """Transformations for wafer bin maps from WM-811K."""
    def __init__(self,
                 size: tuple = (96, 96),
                 mode: str = 'test',
                 **kwargs):

        if isinstance(size, int):
            size = (size, size)
        defaults = dict(size=size, mode=mode)
        defaults.update(kwargs)   # Augmentation-specific arguments are configured here.
        self.defaults = defaults  # Falls back to default values if not specified.

        if mode == 'crop':
            transform = self.crop_transform(**defaults)
        elif mode == 'cutout':
            transform = self.cutout_transform(**defaults)
        elif mode == 'noise':
            transform = self.noise_transform(**defaults)
        elif mode == 'rotate':
            transform = self.rotate_transform(**defaults)
        elif mode == 'shift':
            transform = self.shift_transform(**defaults)
        elif mode == 'test':
            transform = self.test_transform(**defaults)
        elif mode in ['crop+cutout', 'cutout+crop']:
            transform = self.crop_cutout_transform(**defaults)
        elif mode in ['crop+noise', 'noise+crop']:
            transform = self.crop_noise_transform(**defaults)
        elif mode in ['crop+rotate', 'rotate+crop']:
            transform = self.crop_rotate_transform(**defaults)
        elif mode in ['crop+shift', 'shift+crop']:
            transform = self.crop_shift_transform(**defaults)
        elif mode in ['cutout+noise', 'noise+cutout']:
            transform = self.cutout_noise_transform(**defaults)
        elif mode in ['cutout+rotate', 'rotate+cutout']:
            transform = self.cutout_rotate_transform(**defaults)
        elif mode in ['cutout+shift', 'shift+cutout']:
            transform = self.cutout_shift_transform(**defaults)
        elif mode in ['noise+rotate', 'rotate+noise']:
            transform = self.noise_rotate_transform(**defaults)
        elif mode in ['noise+shift', 'shift+noise']:
            transform = self.noise_shift_transform(**defaults)
        elif mode in ['rotate+shift', 'shift+rotate']:
            transform = self.rotate_shift_transform(**defaults)
        else:
            raise NotImplementedError

        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']

    def __repr__(self):
        repr_str = self.__class__.__name__
        for k, v in self.defaults.items():
            repr_str += f"\n{k}: {v}"
        return repr_str

    @staticmethod
    def crop_transform(size: tuple, scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1), **kwargs) -> list:  # pylint: disable=unused-argument
        """
        Crop-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_transform(size: tuple, num_holes: int = 4, cut_ratio: float = 0.2, **kwargs) -> list:  # pylint: disable=unused-argument
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM()
        ]

        return transform

    @staticmethod
    def noise_transform(size: tuple, noise: float = 0.05, **kwargs) -> list:  # pylint: disable=unused-argument
        if noise <= 0.:
            raise ValueError("'noise' probability must be larger than zero.")
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def rotate_transform(size: tuple, **kwargs) -> list:  # pylint: disable=unused-argument
        """
        Rotation-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def shift_transform(size: tuple, shift: float = 0.25, **kwargs) -> list:  # pylint: disable=unused-argument
        transform = [
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def test_transform(size: tuple, **kwargs) -> list:  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_cutout_transform(size: tuple,
                              scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                              num_holes: int = 4, cut_ratio: float = 0.2,
                              **kwargs) -> list:
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_noise_transform(size: tuple, # pylint: disable=unused-argument
                             scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                             noise: float = 0.05,
                             **kwargs) -> list:
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def crop_rotate_transform(size: tuple, scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1), **kwargs) -> list: # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def crop_shift_transform(size: tuple,  # pylint: disable=unused-argument
                             scale: tuple = (0.5, 1.0), ratio: tuple = (0.9, 1.1),
                             shift: float = 0.25,
                             **kwargs) -> list:
        transform = [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_noise_transform(size: tuple,
                               num_holes: int = 4, cut_ratio: float = 0.2,
                               noise: float =0.05,
                               **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def cutout_rotate_transform(size: tuple,
                                num_holes: int = 4, cut_ratio: float = 0.2,
                                **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def cutout_shift_transform(size: tuple,
                               num_holes: int = 4, cut_ratio: float = 0.2,
                               shift: float = 0.25,
                               **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def noise_rotate_transform(size: tuple, noise: float = 0.05, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def noise_shift_transform(size: tuple, noise: float = 0.05, shift: float = 0.25, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

        return transform

    @staticmethod
    def rotate_shift_transform(size: tuple, shift: float = 0.25, **kwargs):  # pylint: disable=unused-argument
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            ToWBM(),
        ]

        return transform
