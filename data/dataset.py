"""Dataset classes for cell generation."""
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tifffile import imread
from torchvision import transforms


class RandomHorizontalFlip(torch.nn.Module):
    """Randomly flip images horizontally with given probability."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[0])
        return x


class RandomVerticalFlip(torch.nn.Module):
    """Randomly flip images vertically with given probability."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(x, dims=[1])
        return x


class FullFieldDataset(Dataset):
    """
    Dataset for full-field microscopy images with protein and cell line labels.

    Loads 4-channel TIFF images and splits them into:
    - Ground truth image (channel 1): The target protein channel
    - Conditioning image (channels 0, 2, 3): Reference channels

    Also provides CLIP-preprocessed images for conditioning.

    Args:
        data_root: Root directory containing the image files.
        train_csv: Path to CSV file with training image filenames.
        test_csv: Path to CSV file with test image filenames.
        cellline_map: Path to pickle file mapping cell line names to indices.
        antibody_map: Path to pickle file mapping antibody names to indices.
        is_train: Whether this is a training dataset (enables augmentation).
        data_len: Optional limit on dataset size (-1 for no limit).
        image_size: Target image size [H, W].
    """

    def __init__(
        self,
        data_root: str,
        train_csv: str,
        test_csv: str,
        cellline_map: str,
        antibody_map: str,
        is_train: bool = True,
        data_len: int = -1,
        image_size: List[int] = [256, 256],
    ):
        # Load file list from CSV
        if is_train:
            flist = pd.read_csv(train_csv, header=0)
            flist = flist['train_images'].tolist()
        else:
            flist = pd.read_csv(test_csv, header=0)
            flist = flist['test_images'].tolist()

        # Optionally limit dataset size
        if data_len > 0:
            idx = np.random.choice(len(flist), data_len, replace=False)
            self.flist = [flist[i] for i in idx]
        else:
            self.flist = flist

        # Transforms
        self.tfs = transforms.Compose([
            torch.from_numpy,
            RandomHorizontalFlip(p=0.25),
            RandomVerticalFlip(p=0.25),
        ])
        self.tfs_no_flip = transforms.Compose([
            torch.from_numpy,
        ])

        self.is_train = is_train
        self.img_shape = image_size
        self.data_root = data_root

        # Load label dictionaries
        with open(cellline_map, "rb") as f:
            self.cell_line_dict = pickle.load(f)
        with open(antibody_map, "rb") as f:
            self.label_dict = pickle.load(f)

    def __len__(self) -> int:
        return len(self.flist)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and preprocess a single sample.

        Returns:
            dict with keys:
                - gt_image: Ground truth image [1, H, W]
                - cond_image: Conditioning image [3, H, W]
                - image: Full image [4, H, W]
                - label: Protein label index
                - protein_name: Protein/antibody name
                - cell_line: Cell line index
                - cell_line_name: Cell line name
                - path: File path
                - clip_image: CLIP-preprocessed image [3, 224, 224]
        """
        file_name = str(self.flist[idx]).zfill(5)
        img = imread(f'{self.data_root}/{file_name}')

        # Apply transforms
        if self.is_train:
            img = self.tfs(img)
        else:
            img = self.tfs_no_flip(img)

        # Split channels
        gt_img = img[:, :, [1]]  # Protein channel
        cond_img = img[:, :, [0, 2, 3]]  # Reference channels

        # Convert to channel-first format
        gt_img = torch.permute(gt_img, (2, 0, 1))
        cond_img = torch.permute(cond_img, (2, 0, 1))
        
        # Parse cell line and antibody from filename
        cell_line = file_name.split('_')[0]
        ab = file_name.split('_')[1]

        if cell_line in self.cell_line_dict:
            cell_line_ = self.cell_line_dict[cell_line]
        else:
            cell_line_ = 0

        return {
            'gt_image': gt_img,
            'cond_image': cond_img,
            'label': self.label_dict[ab],
            'protein_name': ab,
            'cell_line': cell_line_,
            'cell_line_name': cell_line,
            'path': file_name,
        }
