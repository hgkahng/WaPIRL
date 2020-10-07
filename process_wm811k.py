# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from sklearn.model_selection import train_test_split
from utils.logging import get_tqdm_config


class WM811kProcessor(object):
    def __init__(self, wm811k_file: str):

        start_time = time.time()
        self.data = pd.read_pickle(wm811k_file)
        print(f'Successively loaded WM811k data. {time.time() - start_time:.2f}s')

        self.data['labelString'] = self.data['failureType'].apply(self.getLabelString)           # ..., '-'
        self.data['trainTestLabel'] = self.data['trianTestLabel'].apply(self.getTrainTestLabel)  # -1, 0, 1

        self.data['waferMapDim'] = self.data['waferMap'].apply(lambda x: x.shape)
        self.data['waferMapSize'] = self.data['waferMapDim'].apply(lambda x: x[0] * x[1])
        self.data['lotName'] = self.data['lotName'].apply(lambda x: x.replace('lot', ''))
        self.data['waferIndex'] = self.data['waferIndex'].astype(int)

    @staticmethod
    def save_image(arr: np.ndarray, filepath: str = 'image.png', vmin: int = 0, vmax: int = 2):
        scaled_arr = (arr / vmax) * 255
        img = Image.fromarray(scaled_arr.astype(np.uint8))
        img.save(filepath, dpi=(500, 500))

    @staticmethod
    def load_image(filepath: str = 'image.png'):
        return Image.open(filepath)

    def write_images(self, root: str, indices: list or tuple):
        """Write wafer images to .png files."""
        os.makedirs(root, exist_ok=True)
        with tqdm.tqdm(**get_tqdm_config(total=len(indices), leave=True, color='yellow')) as pbar:
            for i, row in self.data.loc[indices].iterrows():
                pngfile = os.path.join(root, row['labelString'], f'{i:06}.png')
                os.makedirs(os.path.dirname(pngfile), exist_ok=True)
                self.save_image(row['waferMap'], pngfile)
                pbar.set_description_str(f" {root} - {i:06} ")
                pbar.update(1)

    def write_unlabeled_images(self,
                               root: str = './data/wm811k/unlabeled/',
                               train_size: float = 0.8,
                               valid_size: float = 0.1):
        """Write wafer images without labels."""
        test_size = 1 - train_size - valid_size

        # Get train / validation / test indices
        unlabeled_indices = self.data.loc[self.data['trainTestLabel'] == -1].index
        train_indices, temp_indices = train_test_split(
            unlabeled_indices,
            train_size=train_size,
            shuffle=True,
            random_state=2015010720,
        )
        valid_indices, test_indices = train_test_split(
            temp_indices,
            train_size=valid_size / (valid_size + test_size),
            shuffle=True,
            random_state=2015010720,
        )

        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    def write_labeled_images(self,
                             root: str = './data/wm811k/labeled/',
                             train_size: float = 0.8,
                             valid_size: float = 0.1):
        """Write wafer images with labels."""
        test_size = 1 - train_size - valid_size

        labeled_indices = self.data.loc[self.data['trainTestLabel'] != -1].index
        temp_indices, test_indices = train_test_split(
            labeled_indices,
            test_size=test_size,
            stratify=self.data.loc[labeled_indices, 'labelString'],
            shuffle=True,
            random_state=2015010720,
        )
        train_indices, valid_indices = train_test_split(
            temp_indices,
            test_size=valid_size/(train_size + valid_size),
            stratify=self.data.loc[temp_indices, 'labelString'],
            random_state=2015010720,
        )

        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    @staticmethod
    def nearest_interpolate(arr, s=(40, 40)):
        assert isinstance(arr, np.ndarray) and len(arr.shape) == 2
        ptt = torch.from_numpy(arr).view(1, 1, *arr.shape).float()
        return F.interpolate(ptt, size=s, mode='nearest').squeeze().long().numpy()

    @staticmethod
    def getLabelString(x):
        if len(x) == 1:
            ls = x[0][0].strip().lower()  # Labeled (9 classes)
        else:
            ls = '-'
        return ls

    @staticmethod
    def getTrainTestLabel(x):
        d = {
            'unlabeled': -1,  # 638,507
            'training': 0,    # 118,595
            'test': 1,        #  54,355
        }
        if len(x) == 1:
            lb = x[0][0].strip().lower()
        else:
            lb = 'unlabeled'
        return d[lb]


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""

        parser = argparse.ArgumentParser("Process WM-811k data to individual image files.", add_help=True)
        parser.add_argument('--labeled_root', type=str, default='./data/wm811k/labeled')
        parser.add_argument('--unlabeled_root', type=str, default='./data/wm811k/unlabeled')
        parser.add_argument('--labeled_train_size', type=float, default=0.8)
        parser.add_argument('--labeled_valid_size', type=float, default=0.1)
        parser.add_argument('--unlabeled_train_size', type=float, default=0.8)
        parser.add_argument('--unlabeled_valid_size', type=float, default=0.1)

        return parser.parse_args()

    def check_files_exist_in_directory(directory: str, file_ext: str = 'png', recursive: bool = True):
        """Check existence of files of specific types are under a directory"""
        files = glob.glob(os.path.join(directory, f"**/*.{file_ext}"), recursive=recursive)
        return len(files) > 0  # True if files exist, else False.

    args = parse_args()
    processor = WM811kProcessor(wm811k_file='./data/wm811k/LSWMD.pkl')

    if not check_files_exist_in_directory(args.labeled_root):
        processor.write_labeled_images(root='./data/wm811k/labeled/', train_size=0.8, valid_size=0.1)
    else:
        print(f"Labeled images exist in `{args.labeled_root}`. Skipping...")

    if not check_files_exist_in_directory(args.unlabeled_root):
        processor.write_unlabeled_images(root='./data/wm811k/unlabeled/', train_size=0.8, valid_size=0.1)
    else:
        print(f"Unlabeled images exist in `{args.unlabeled_root}`. Skipping...")
