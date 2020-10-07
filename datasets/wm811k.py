# -*- coding: utf-8 -*-

import os
import glob
import pathlib

import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class WM811K(Dataset):
    label2idx = {
        'center'    : 0,
        'donut'     : 1,
        'edge-loc'  : 2,
        'edge-ring' : 3,
        'loc'       : 4,
        'random'    : 5,
        'scratch'   : 6,
        'near-full' : 7,
        'none'      : 8,
        '-'         : 9,
    }
    idx2label = [k for k in label2idx.keys()]
    num_classes = len(idx2label) - 1  # exclude unlabeled (-)

    def __init__(self, root, transform=None, proportion=1.0, decouple_input: bool = True, **kwargs):
        super(WM811K, self).__init__()

        self.root = root
        self.transform = transform
        self.proportion = proportion
        self.decouple_input = decouple_input

        images  = sorted(glob.glob(os.path.join(root, '**/*.png'), recursive=True))  # Get paths to images
        labels  = [pathlib.PurePath(image).parent.name for image in images]          # Parent directory names are class label strings
        targets = [self.label2idx[l] for l in labels]                                # Convert class label strings to integer target values
        samples = list(zip(images, targets))                                         # Make (path, target) pairs

        if self.proportion < 1.0:
            # Randomly sample a proportion of the data
            self.samples, _ = train_test_split(
                samples,
                train_size=self.proportion,
                stratify=[s[1] for s in samples],
                shuffle=True,
                random_state=1993 + kwargs.get('seed', 0),
            )
        else:
            self.samples = samples

    def __getitem__(self, idx):

        path, y = self.samples[idx]
        x = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(x)

        if self.decouple_input:
            x = self.decouple_mask(x)

        return dict(x=x, y=y, idx=idx)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL. Use with `torchvision`."""
        return Image.open(filepath)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)

    @staticmethod
    def decouple_mask(x: torch.Tensor):
        """
        Decouple input with existence mask.
        Defect bins = 2, Normal bins = 1, Null bins = 0
        """
        m = x.gt(0).float()
        x = torch.clamp(x - 1, min=0., max=1.)

        return torch.cat([x, m], dim=0)


class WM811KForWaPIRL(WM811K):
    def __init__(self, root, transform=None, positive_transform=None, decouple_input: bool = True):
        super(WM811KForWaPIRL, self).__init__(root, transform, proportion=1.0, decouple_input=decouple_input)
        self.positive_transform = positive_transform

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = self.load_image_cv2(path)

        if self.transform is not None:
            x = self.transform(img)

        if self.positive_transform is not None:
            x_t = self.positive_transform(img)

        if self.decouple_input:
            x = self.decouple_mask(x)
            x_t = self.decouple_mask(x_t)

        return dict(x=x, x_t=x_t, y=y, idx=idx)
