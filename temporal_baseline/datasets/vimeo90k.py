"""Vimeo-90K Septuplet dataset for video super-resolution.

Expected directory layout (after downloading from http://toflow.csail.mit.edu/):

    data/vimeo_septuplet/
    ├── sequences/
    │   ├── 00001/
    │   │   ├── 0001/
    │   │   │   ├── im1.png  ...  im7.png   (GT, 448x256)
    │   │   │   ...
    │   ...
    ├── sep_trainlist.txt
    └── sep_testlist.txt

LR frames are generated on-the-fly via bicubic downsampling (scale=4).
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
    """

    def __init__(self, root, split='train', scale=4, patch_size=None):
        """
        Args:
            root: path to vimeo_septuplet/ directory.
            split: 'train' or 'test'.
            scale: downsampling factor (default 4).
            patch_size: if set, randomly crop GT to (patch_size, patch_size)
                        and LR to (patch_size//scale, patch_size//scale).
                        Only used for training.
        """
        super().__init__()
        self.root = Path(root)
        self.scale = scale
        self.patch_size = patch_size
        self.split = split

        list_file = self.root / ('sep_trainlist.txt' if split == 'train' else 'sep_testlist.txt')
        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()  # [0,1] float CHW

    def __len__(self):
        return len(self.sequences)

    def _load_frames(self, seq_path):
        frames = []
        for i in range(1, 8):
            img = Image.open(seq_path / f'im{i}.png').convert('RGB')
            frames.append(img)
        return frames

    def _downsample(self, img_pil):
        w, h = img_pil.size
        return img_pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)

    def __getitem__(self, idx):
        seq_path = self.root / 'sequences' / self.sequences[idx]
        gt_frames = self._load_frames(seq_path)

        if self.patch_size and self.split == 'train':
            gt_frames, crop_params = self._random_crop(gt_frames)

        lr_frames = [self._downsample(f) for f in gt_frames]

        # convert to tensors
        gt_tensors = torch.stack([self.to_tensor(f) for f in gt_frames])  # (7, 3, H, W)
        lr_tensors = torch.stack([self.to_tensor(f) for f in lr_frames])  # (7, 3, H/s, W/s)

        return {'lr': lr_tensors, 'gt': gt_tensors, 'seq_name': self.sequences[idx]}

    def _random_crop(self, frames):
        w, h = frames[0].size
        ps = self.patch_size
        x = np.random.randint(0, w - ps + 1)
        y = np.random.randint(0, h - ps + 1)
        return [f.crop((x, y, x + ps, y + ps)) for f in frames], (x, y)


class Vimeo90KTriplet(Dataset):
    """Vimeo-90K triplet dataset (3 frames). Predict middle frame from 3 neighbors.

    Layout:
        data/vimeo_triplet/
        ├── sequences/  (im1.png, im2.png, im3.png per folder)
        ├── tri_trainlist.txt
        └── tri_testlist.txt
    """

    def __init__(self, root, split='train', scale=4, patch_size=None):
        super().__init__()
        self.root = Path(root)
        self.scale = scale
        self.patch_size = patch_size
        self.split = split

        list_file = self.root / ('tri_trainlist.txt' if split == 'train' else 'tri_testlist.txt')
        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f if line.strip()]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.root / 'sequences' / self.sequences[idx]
        frames = []
        for i in range(1, 4):
            img = Image.open(seq_path / f'im{i}.png').convert('RGB')
            frames.append(img)

        if self.patch_size and self.split == 'train':
            w, h = frames[0].size
            ps = self.patch_size
            x = np.random.randint(0, w - ps + 1)
            y = np.random.randint(0, h - ps + 1)
            frames = [f.crop((x, y, x + ps, y + ps)) for f in frames]

        lr_frames = [f.resize((f.size[0] // self.scale, f.size[1] // self.scale), Image.BICUBIC) for f in frames]

        gt_tensors = torch.stack([self.to_tensor(f) for f in frames])   # (3, 3, H, W)
        lr_tensors = torch.stack([self.to_tensor(f) for f in lr_frames])  # (3, 3, h, w)

        return {'lr': lr_tensors, 'gt': gt_tensors, 'seq_name': self.sequences[idx]}
