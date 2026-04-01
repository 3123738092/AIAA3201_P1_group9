"""Inference & Evaluation for Part 1 baselines on Vimeo-90K or REDS.

Methods:
    1. bicubic      — Bicubic interpolation
    2. lanczos      — Lanczos interpolation
    3. srcnn        — SRCNN (requires trained checkpoint)
    4. temporal_avg — Multi-frame weighted averaging + optional Unsharp Masking

Usage:
    # Bicubic on REDS val
    python inference.py --dataset reds --method bicubic --data_root /path/to/reds

    # SRCNN on Vimeo
    python inference.py --dataset vimeo --method srcnn --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

    # Temporal averaging + USM on REDS4
    python inference.py --dataset reds4 --method temporal_avg --data_root /path/to/reds --base_method srcnn --checkpoint checkpoints/srcnn_best.pth --unsharp
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets.vimeo90k import Vimeo90KSeptuplet
from datasets.reds import REDSDataset, REDS4Dataset
from models.srcnn import SRCNN
from utils.metrics import calc_psnr, calc_ssim


# ───────────────────── Spatial upsampling methods ─────────────────────

def upsample_bicubic(lr_np, scale):
    h, w = lr_np.shape[:2]
    return cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def upsample_lanczos(lr_np, scale):
    h, w = lr_np.shape[:2]
    return cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def upsample_srcnn(lr_np, scale, model, device):
    h, w = lr_np.shape[:2]
    lr_up = cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(lr_up).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        out = model(tensor).clamp(0, 1)
    return (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ───────────────────── Temporal fusion ─────────────────────

def temporal_weighted_average(frames, center_idx, window=3):
    """Gaussian-weighted average of neighboring frames around center_idx."""
    half = window // 2
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)

    weights = []
    selected = []
    for i in range(start, end):
        dist = abs(i - center_idx)
        w = np.exp(-0.5 * dist ** 2)
        weights.append(w)
        selected.append(frames[i].astype(np.float64))

    weights = np.array(weights) / sum(weights)
    result = sum(w * f for w, f in zip(weights, selected))
    return np.clip(result, 0, 255).astype(np.uint8)


def unsharp_mask(img, sigma=1.0, strength=0.5):
    """USM(x) = x + strength * (x - GaussianBlur(x))"""
    blurred = cv2.GaussianBlur(img.astype(np.float64), (0, 0), sigma)
    sharpened = img.astype(np.float64) + strength * (img.astype(np.float64) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ───────────────────── Main ─────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='reds',
                   choices=['vimeo', 'reds', 'reds4'],
                   help='vimeo=Vimeo-90K, reds=REDS val, reds4=REDS4 test')
    p.add_argument('--method', type=str, required=True,
                   choices=['bicubic', 'lanczos', 'srcnn', 'temporal_avg'])
    p.add_argument('--base_method', type=str, default='bicubic',
                   choices=['bicubic', 'lanczos', 'srcnn'],
                   help='Base spatial method for temporal_avg')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--num_frames', type=int, default=7, help='Frames per sample (REDS)')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='results')
    p.add_argument('--unsharp', action='store_true')
    p.add_argument('--unsharp_sigma', type=float, default=1.0)
    p.add_argument('--unsharp_strength', type=float, default=0.5)
    p.add_argument('--window', type=int, default=5, help='Temporal averaging window size')
    p.add_argument('--max_sequences', type=int, default=0, help='Limit samples (0=all)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_images', action='store_true')
    return p.parse_args()


def build_dataset(args):
    if args.dataset == 'vimeo':
        return Vimeo90KSeptuplet(args.data_root, split='test', scale=args.scale)
    elif args.dataset == 'reds':
        return REDSDataset(args.data_root, split='val', scale=args.scale,
                           num_frames=args.num_frames)
    elif args.dataset == 'reds4':
        return REDS4Dataset(args.data_root, scale=args.scale,
                            num_frames=args.num_frames)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


def spatial_upsample(lr_np, method, scale, srcnn_model=None, device='cpu'):
    if method == 'bicubic':
        return upsample_bicubic(lr_np, scale)
    elif method == 'lanczos':
        return upsample_lanczos(lr_np, scale)
    elif method == 'srcnn':
        return upsample_srcnn(lr_np, scale, srcnn_model, device)
    else:
        raise ValueError(f'Unknown method: {method}')


def tensor_to_numpy(t):
    """(C, H, W) tensor [0,1] -> HWC uint8 numpy."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()

    # Build output dir name
    method_name = args.method
    if args.method == 'temporal_avg':
        method_name = f'temporal_avg_{args.base_method}_w{args.window}'
        if args.unsharp:
            method_name += '_usm'
    output_dir = Path(args.output_dir) / args.dataset / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SRCNN if needed
    srcnn_model = None
    effective_method = args.method if args.method != 'temporal_avg' else args.base_method
    if effective_method == 'srcnn':
        assert args.checkpoint, 'Must provide --checkpoint for SRCNN'
        srcnn_model = SRCNN().to(args.device)
        srcnn_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        srcnn_model.eval()

    # Dataset
    dataset = build_dataset(args)
    if args.max_sequences > 0:
        if hasattr(dataset, 'sequences'):
            dataset.sequences = dataset.sequences[:args.max_sequences]
        elif hasattr(dataset, 'samples'):
            dataset.samples = dataset.samples[:args.max_sequences]

    # Metrics
    psnr_all, ssim_all = [], []
    device = args.device

    print(f'Dataset: {args.dataset} | Method: {method_name} | Samples: {len(dataset)} | Scale: x{args.scale}')

    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[idx]
        lr_frames = sample['lr']   # (T, 3, h, w)
        gt_frames = sample['gt']   # (T, 3, H, W)
        seq_name = sample['seq_name']
        T = lr_frames.shape[0]
        center = T // 2

        # Spatial upsample all frames
        sr_frames = []
        for t in range(T):
            lr_np = tensor_to_numpy(lr_frames[t])
            sr_np = spatial_upsample(lr_np, effective_method, args.scale, srcnn_model, device)
            sr_frames.append(sr_np)

        # Temporal fusion
        if args.method == 'temporal_avg':
            fused = []
            for t in range(len(sr_frames)):
                avg = temporal_weighted_average(sr_frames, t, window=args.window)
                if args.unsharp:
                    avg = unsharp_mask(avg, sigma=args.unsharp_sigma, strength=args.unsharp_strength)
                fused.append(avg)
            sr_frames = fused

        # Evaluate center frame
        gt_np = tensor_to_numpy(gt_frames[center])
        sr_np = sr_frames[center]

        psnr_all.append(calc_psnr(sr_np, gt_np))
        ssim_all.append(calc_ssim(sr_np, gt_np))

        # Save images
        if args.save_images:
            seq_out = output_dir / seq_name.replace('/', '_')
            seq_out.mkdir(parents=True, exist_ok=True)
            for t, f in enumerate(sr_frames):
                Image.fromarray(f).save(seq_out / f'{t:08d}.png')

    avg_psnr = np.mean(psnr_all)
    avg_ssim = np.mean(ssim_all)
    print(f'\n===== Results: {args.dataset} / {method_name} =====')
    print(f'  PSNR: {avg_psnr:.2f} dB')
    print(f'  SSIM: {avg_ssim:.4f}')
    print(f'  Evaluated on {len(psnr_all)} samples (center frame)')

    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Method: {method_name}\n')
        f.write(f'Scale: x{args.scale}\n')
        f.write(f'PSNR: {avg_psnr:.2f}\n')
        f.write(f'SSIM: {avg_ssim:.4f}\n')
        f.write(f'Samples: {len(psnr_all)}\n')


if __name__ == '__main__':
    main()
