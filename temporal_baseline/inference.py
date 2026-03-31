"""Inference & Evaluation for Part 1 baselines on Vimeo-90K.

Methods:
    1. bicubic      — Bicubic interpolation
    2. lanczos      — Lanczos interpolation
    3. srcnn        — SRCNN (requires trained checkpoint)
    4. temporal_avg — Multi-frame weighted averaging + optional Unsharp Masking

Usage:
    # Bicubic baseline
    python inference.py --method bicubic --data_root ../../data/vimeo_septuplet

    # SRCNN
    python inference.py --method srcnn --data_root ../../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

    # Temporal averaging (on top of bicubic)
    python inference.py --method temporal_avg --data_root ../../data/vimeo_septuplet --unsharp

    # Temporal averaging (on top of SRCNN)
    python inference.py --method temporal_avg --data_root ../../data/vimeo_septuplet --base_method srcnn --checkpoint checkpoints/srcnn_best.pth --unsharp
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
from models.srcnn import SRCNN
from utils.metrics import calc_psnr, calc_ssim, LPIPSMetric


# ───────────────────── Spatial upsampling methods ─────────────────────

def upsample_bicubic(lr_np, scale):
    """lr_np: HWC uint8 -> HWC uint8."""
    h, w = lr_np.shape[:2]
    return cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def upsample_lanczos(lr_np, scale):
    h, w = lr_np.shape[:2]
    return cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def upsample_srcnn(lr_np, scale, model, device):
    """Bicubic upsample + SRCNN refinement."""
    h, w = lr_np.shape[:2]
    lr_up = cv2.resize(lr_np, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(lr_up).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        out = model(tensor).clamp(0, 1)
    return (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ───────────────────── Temporal fusion ─────────────────────

def temporal_weighted_average(frames, center_idx, window=3):
    """Weighted average of neighboring frames around center_idx.

    Gaussian-like weights: closer frames get higher weight.
    frames: list of HWC uint8 arrays (all same size, already upscaled).
    """
    half = window // 2
    start = max(0, center_idx - half)
    end = min(len(frames), center_idx + half + 1)

    weights = []
    selected = []
    for i in range(start, end):
        dist = abs(i - center_idx)
        w = np.exp(-0.5 * dist ** 2)  # sigma=1 Gaussian
        weights.append(w)
        selected.append(frames[i].astype(np.float64))

    weights = np.array(weights) / sum(weights)
    result = sum(w * f for w, f in zip(weights, selected))
    return np.clip(result, 0, 255).astype(np.uint8)


def unsharp_mask(img, sigma=1.0, strength=0.5):
    """Apply Unsharp Masking to enhance high-frequency edges.

    USM(x) = x + strength * (x - GaussianBlur(x))
    """
    blurred = cv2.GaussianBlur(img.astype(np.float64), (0, 0), sigma)
    sharpened = img.astype(np.float64) + strength * (img.astype(np.float64) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ───────────────────── Main ─────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, required=True,
                   choices=['bicubic', 'lanczos', 'srcnn', 'temporal_avg'])
    p.add_argument('--base_method', type=str, default='bicubic',
                   choices=['bicubic', 'lanczos', 'srcnn'],
                   help='Base spatial method for temporal_avg')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--checkpoint', type=str, default=None, help='SRCNN checkpoint path')
    p.add_argument('--output_dir', type=str, default='../../results/temporal_baseline')
    p.add_argument('--unsharp', action='store_true', help='Apply Unsharp Masking after temporal avg')
    p.add_argument('--unsharp_sigma', type=float, default=1.0)
    p.add_argument('--unsharp_strength', type=float, default=0.5)
    p.add_argument('--window', type=int, default=5, help='Temporal averaging window size')
    p.add_argument('--max_sequences', type=int, default=0, help='Limit sequences for quick test (0=all)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_images', action='store_true', help='Save output images')
    return p.parse_args()


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
    output_dir = Path(args.output_dir) / args.method
    if args.method == 'temporal_avg':
        suffix = f'{args.base_method}_w{args.window}'
        if args.unsharp:
            suffix += '_usm'
        output_dir = Path(args.output_dir) / f'temporal_avg_{suffix}'
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
    dataset = Vimeo90KSeptuplet(args.data_root, split='test', scale=args.scale)
    if args.max_sequences > 0:
        dataset.sequences = dataset.sequences[:args.max_sequences]

    # Metrics
    psnr_all, ssim_all = [], []
    device = args.device

    print(f'Method: {args.method}  |  Sequences: {len(dataset)}  |  Scale: x{args.scale}')

    for idx in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[idx]
        lr_frames = sample['lr']   # (7, 3, h, w)
        gt_frames = sample['gt']   # (7, 3, H, W)
        seq_name = sample['seq_name']

        # Spatial upsample all frames
        sr_frames = []
        for t in range(lr_frames.shape[0]):
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

        # Evaluate center frame (index 3)
        center = 3
        gt_np = tensor_to_numpy(gt_frames[center])
        sr_np = sr_frames[center]

        psnr_all.append(calc_psnr(sr_np, gt_np))
        ssim_all.append(calc_ssim(sr_np, gt_np))

        # Save images
        if args.save_images:
            seq_out = output_dir / seq_name.replace('/', '_')
            seq_out.mkdir(parents=True, exist_ok=True)
            for t, f in enumerate(sr_frames):
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.shape[-1] == 3 else f).save(
                    seq_out / f'im{t + 1}.png'
                )
                # Actually, the frame is already RGB from PIL loading
                Image.fromarray(f).save(seq_out / f'im{t + 1}.png')

    avg_psnr = np.mean(psnr_all)
    avg_ssim = np.mean(ssim_all)
    print(f'\n===== Results: {args.method} =====')
    print(f'  PSNR: {avg_psnr:.2f} dB')
    print(f'  SSIM: {avg_ssim:.4f}')
    print(f'  Evaluated on {len(psnr_all)} sequences (center frame)')

    # Save results to txt
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f'Method: {args.method}\n')
        f.write(f'Scale: x{args.scale}\n')
        f.write(f'PSNR: {avg_psnr:.2f}\n')
        f.write(f'SSIM: {avg_ssim:.4f}\n')
        f.write(f'Sequences: {len(psnr_all)}\n')


if __name__ == '__main__':
    main()
