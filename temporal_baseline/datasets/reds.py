"""REDS dataset for video super-resolution.

Expected layout after unzipping:

    data/reds/
    ├── train_sharp/          000~239, each has 00000000.png ~ 00000099.png (GT)
    ├── train_sharp_bicubic/
    │   └── X4/               000~239, same naming (LR, 4x downsampled)
    ├── val_sharp/            000~029 (GT)
    └── val_sharp_bicubic/
        └── X4/               000~029 (LR)

Each sequence has 100 frames: 00000000.png ... 00000099.png
GT resolution: 720x1280, LR resolution: 180x320
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class REDSDataset(Dataset):
    """REDS dataset for video super-resolution.

    Each sample returns `num_frames` consecutive LR/GT frame pairs centered at a given frame.
    """

    def __init__(self, root, split='train', scale=4, num_frames=7, patch_size=None):
        """
        Args:
            root: path to reds/ directory (parent of train_sharp/, etc.)
            split: 'train' or 'val'.
            scale: downsampling factor (default 4).
            num_frames: number of consecutive frames per sample (default 7, matching Vimeo).
            patch_size: if set, randomly crop GT. Training only.
        """
        super().__init__()
        self.root = Path(root)
        self.scale = scale
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.split = split

        if split == 'train':
            self.gt_dir = self.root / 'train_sharp'
            self.lr_dir = self.root / 'train_sharp_bicubic' / 'X4'
        elif split == 'val':
            self.gt_dir = self.root / 'val_sharp'
            self.lr_dir = self.root / 'val_sharp_bicubic' / 'X4'
        else:
            raise ValueError(f'Unknown split: {split}')

        # List all sequences
        self.sequences = sorted([
            d.name for d in self.gt_dir.iterdir() if d.is_dir()
        ])

        # Build (seq, center_frame) index
        # Each sequence has 100 frames; we need num_frames//2 margin on each side
        self.samples = []
        half = num_frames // 2
        for seq in self.sequences:
            n_frames = len(list((self.gt_dir / seq).glob('*.png')))
            for t in range(half, n_frames - half):
                self.samples.append((seq, t))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _frame_path(self, base_dir, seq, frame_idx):
        return base_dir / seq / f'{frame_idx:08d}.png'

    def __getitem__(self, idx):
        seq, center = self.samples[idx]
        half = self.num_frames // 2

        gt_frames = []
        lr_frames = []
        for t in range(center - half, center + half + 1):
            gt_img = Image.open(self._frame_path(self.gt_dir, seq, t)).convert('RGB')
            lr_img = Image.open(self._frame_path(self.lr_dir, seq, t)).convert('RGB')

            if self.patch_size and self.split == 'train':
                gt_img, lr_img = self._random_crop_pair(gt_img, lr_img)

            gt_frames.append(self.to_tensor(gt_img))
            lr_frames.append(self.to_tensor(lr_img))

        gt_tensors = torch.stack(gt_frames)  # (num_frames, 3, H, W)
        lr_tensors = torch.stack(lr_frames)  # (num_frames, 3, h, w)

        return {
            'lr': lr_tensors,
            'gt': gt_tensors,
            'seq_name': f'{seq}/{center:08d}'
        }

    def _random_crop_pair(self, gt_img, lr_img):
        """Randomly crop GT and LR with consistent coordinates."""
        gt_w, gt_h = gt_img.size
        ps = self.patch_size
        lr_ps = ps // self.scale

        x = np.random.randint(0, gt_w - ps + 1)
        y = np.random.randint(0, gt_h - ps + 1)
        lx, ly = x // self.scale, y // self.scale

        gt_crop = gt_img.crop((x, y, x + ps, y + ps))
        lr_crop = lr_img.crop((lx, ly, lx + lr_ps, ly + lr_ps))
        return gt_crop, lr_crop


class REDS4Dataset(Dataset):
    """REDS4 test set: sequences 000, 011, 015, 020 from training set.

    Used as the standard test benchmark (GT is available).
    """

    REDS4_SEQS = ['000', '011', '015', '020']

    def __init__(self, root, scale=4, num_frames=7):
        super().__init__()
        self.root = Path(root)
        self.scale = scale
        self.num_frames = num_frames

        self.gt_dir = self.root / 'train_sharp'
        self.lr_dir = self.root / 'train_sharp_bicubic' / 'X4'

        self.samples = []
        half = num_frames // 2
        for seq in self.REDS4_SEQS:
            n_frames = len(list((self.gt_dir / seq).glob('*.png')))
            for t in range(half, n_frames - half):
                self.samples.append((seq, t))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, center = self.samples[idx]
        half = self.num_frames // 2

        gt_frames, lr_frames = [], []
        for t in range(center - half, center + half + 1):
            gt_path = self.gt_dir / seq / f'{t:08d}.png'
            lr_path = self.lr_dir / seq / f'{t:08d}.png'
            gt_frames.append(self.to_tensor(Image.open(gt_path).convert('RGB')))
            lr_frames.append(self.to_tensor(Image.open(lr_path).convert('RGB')))

        return {
            'lr': torch.stack(lr_frames),
            'gt': torch.stack(gt_frames),
            'seq_name': f'{seq}/{center:08d}'
        }
