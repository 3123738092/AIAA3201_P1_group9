"""Vimeo-90K dataset for video super-resolution.

Supports two layouts:

1. Septuplet (full dataset, LR generated on-the-fly):
    vimeo_septuplet/
    ├── sequences/00001/0001/im{1..7}.png   (GT, 448x256)
    ├── sep_trainlist.txt
    └── sep_testlist.txt

2. SR Test Set (pre-computed LR/GT pairs):
    vimeo_super_resolution_test/
    ├── target/00001/0001/im{1..7}.png      (GT)
    ├── low_resolution/00001/0001/im{1..7}.png  (LR, already 4x downsampled)
    ├── sep_testlist.txt
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Vimeo90KSeptuplet(Dataset):
    """Vimeo-90K septuplet dataset.

    Each sample returns 7 LR frames and 7 HR (GT) frames.
    Supports both full septuplet (on-the-fly downsample) and SR test set (pre-computed LR).
    """

    def __init__(self, root, split='train', scale=4, patch_size=None):
        """
        Args:
            root: path to vimeo_septuplet/ or vimeo_super_resolution_test/.
            split: 'train' or 'test'.
            scale: downsampling factor (default 4).
            patch_size: if set, randomly crop GT to (patch_size, patch_size). Training only.
        """
        super().__init__()
        self.root = Path(root)
        self.scale = scale
        self.patch_size = patch_size
        self.split = split

        # Detect layout: SR test set has 'target/' folder, septuplet has 'sequences/'
        if (self.root / 'target').exists():
            self.layout = 'sr_test'
            self.gt_dir = self.root / 'target'
            self.lr_dir = self.root / 'low_resolution'
        else:
            self.layout = 'septuplet'
            self.gt_dir = self.root / 'sequences'
            self.lr_dir = None  # LR generated on-the-fly

        list_file = self.root / ('sep_trainlist.txt' if split == 'train' else 'sep_testlist.txt')
        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sequences)

    def _load_frames(self, folder, num_frames=7):
        frames = []
        for i in range(1, num_frames + 1):
            img = Image.open(folder / f'im{i}.png').convert('RGB')
            frames.append(img)
        return frames

    def _downsample(self, img_pil):
        w, h = img_pil.size
        return img_pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        gt_frames = self._load_frames(self.gt_dir / seq)

        if self.layout == 'sr_test' and self.lr_dir is not None:
            lr_frames_pil = self._load_frames(self.lr_dir / seq)
        else:
            # On-the-fly downsampling for septuplet layout
            if self.patch_size and self.split == 'train':
                gt_frames = self._random_crop(gt_frames)
            lr_frames_pil = [self._downsample(f) for f in gt_frames]

        gt_tensors = torch.stack([self.to_tensor(f) for f in gt_frames])
        lr_tensors = torch.stack([self.to_tensor(f) for f in lr_frames_pil])

        return {'lr': lr_tensors, 'gt': gt_tensors, 'seq_name': seq}

    def _random_crop(self, frames):
        w, h = frames[0].size
        ps = self.patch_size
        x = np.random.randint(0, w - ps + 1)
        y = np.random.randint(0, h - ps + 1)
        return [f.crop((x, y, x + ps, y + ps)) for f in frames]
