"""Shared evaluation script for all VSR methods.

Metrics (matching AIAA3201 Project 1 guideline Section 6.2):
  - PSNR & SSIM       — Pixel Accuracy        (always computed)
  - LPIPS              — Perceptual Quality     (--lpips)
  - FID                — Perceptual Quality     (--fid)
  - tLPIPS             — Temporal Consistency   (--tlpips, requires --save_all_frames in inference)
  - Qualitative        — Visual comparison      (--visualize N)

Reusable across temporal_baseline, real_esrgan, basicvsr, etc.

Usage:
    # Basic: PSNR + SSIM
    python evaluate.py --sr_dir results/vimeo/bicubic \
                       --dataset vimeo --data_root data/vimeo_septuplet

    # All metrics
    python evaluate.py --sr_dir results/vimeo/bicubic \
                       --dataset vimeo --data_root data/vimeo_septuplet \
                       --lpips --fid --tlpips

    # With qualitative visualizations
    python evaluate.py --sr_dir results/vimeo/bicubic \
                       --dataset vimeo --data_root data/vimeo_septuplet \
                       --lpips --fid --tlpips --visualize 10
"""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ───────────────────── Metrics ─────────────────────

def calc_psnr(img1, img2, crop_border=0):
    """PSNR between two HWC uint8 or float64 [0,1] images."""
    from skimage.metrics import peak_signal_noise_ratio
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def calc_ssim(img1, img2, crop_border=0):
    """SSIM between two HWC uint8 or float64 [0,1] images."""
    from skimage.metrics import structural_similarity
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return structural_similarity(img1, img2, channel_axis=-1, data_range=1.0)


class LPIPSMetric:
    """LPIPS perceptual metric wrapper (also used for tLPIPS)."""
    def __init__(self, net='alex', device='cuda'):
        import torch
        import lpips
        self.device = device
        self.fn = lpips.LPIPS(net=net).to(device).eval()

    def _to_tensor(self, img):
        import torch
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return (t * 2.0 - 1.0).to(self.device)

    def calc(self, img1, img2):
        import torch
        with torch.no_grad():
            return self.fn(self._to_tensor(img1), self._to_tensor(img2)).item()

    def calc_tlpips(self, sr_frames, gt_frames):
        """tLPIPS: mean_t LPIPS(SR_t - SR_{t-1}, GT_t - GT_{t-1}).

        Measures temporal consistency by comparing frame-difference patterns.
        """
        import torch
        scores = []
        for i in range(1, len(sr_frames)):
            diff_sr = sr_frames[i].astype(np.float32) - sr_frames[i - 1].astype(np.float32)
            diff_gt = gt_frames[i].astype(np.float32) - gt_frames[i - 1].astype(np.float32)
            # Normalize diffs to [0,1] range for LPIPS input
            diff_sr = np.clip(diff_sr / 510.0 + 0.5, 0, 1)
            diff_gt = np.clip(diff_gt / 510.0 + 0.5, 0, 1)
            scores.append(self.calc(diff_sr, diff_gt))
        return np.mean(scores) if scores else 0.0


# ───────────────────── FID ─────────────────────

def compute_fid(pairs, device, batch_size=50):
    """Compute FID between SR and GT image sets using pytorch-fid."""
    from pytorch_fid.fid_score import calculate_fid_given_paths

    # pytorch-fid expects flat directories; create temp dirs with symlinks
    with tempfile.TemporaryDirectory() as sr_tmp, \
         tempfile.TemporaryDirectory() as gt_tmp:
        for i, pair in enumerate(pairs):
            os.symlink(os.path.abspath(pair['sr']),
                       os.path.join(sr_tmp, f'{i:06d}.png'))
            os.symlink(os.path.abspath(pair['gt']),
                       os.path.join(gt_tmp, f'{i:06d}.png'))

        fid_value = calculate_fid_given_paths(
            [sr_tmp, gt_tmp],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
    return fid_value


# ───────────────────── SR/GT pair discovery ─────────────────────

def find_center_pairs(sr_dir, dataset, data_root):
    """Find center-frame SR/GT pairs (for PSNR, SSIM, LPIPS, FID)."""
    sr_dir = Path(sr_dir)
    data_root = Path(data_root)
    pairs = []

    if dataset == 'vimeo':
        gt_base = (data_root / 'target') if (data_root / 'target').exists() \
            else (data_root / 'sequences')
        lr_base = (data_root / 'low_resolution') if (data_root / 'low_resolution').exists() \
            else None

        for sr_path in sorted(sr_dir.rglob('*.png')):
            rel = sr_path.relative_to(sr_dir)
            # Skip if inside a sequence with multiple frames — only take center (im4)
            parts = rel.parts
            if len(parts) == 3 and parts[2].startswith('im'):
                # Check if this is a multi-frame dir (has im1, im2, ...) or single center
                sibling_count = len(list(sr_path.parent.glob('im*.png')))
                if sibling_count > 1:
                    # Multi-frame dir: only use center frame (im4) for per-image metrics
                    if parts[2] != 'im4.png':
                        continue
            gt_path = gt_base / rel
            if gt_path.exists():
                lr_path = (lr_base / rel) if lr_base else None
                pairs.append({'sr': sr_path, 'gt': gt_path, 'lr': lr_path,
                              'name': str(rel)})

    elif dataset in ('reds', 'reds4'):
        if dataset == 'reds':
            gt_base = data_root / 'val_sharp'
            lr_base = data_root / 'val_sharp_bicubic' / 'X4'
        else:
            gt_base = data_root / 'train_sharp'
            lr_base = data_root / 'train_sharp_bicubic' / 'X4'

        for sr_path in sorted(sr_dir.rglob('*.png')):
            rel = sr_path.relative_to(sr_dir)
            parts = rel.parts
            if len(parts) == 3:
                # Multi-frame structure: seq/center/frame.png — skip non-center
                continue
            gt_path = gt_base / rel
            lr_path = lr_base / rel
            if gt_path.exists():
                pairs.append({'sr': sr_path, 'gt': gt_path,
                              'lr': lr_path if lr_path.exists() else None,
                              'name': str(rel)})

    return pairs


def find_sequences_for_tlpips(sr_dir, dataset, data_root):
    """Find multi-frame sequences for tLPIPS (requires --save_all_frames)."""
    sr_dir = Path(sr_dir)
    data_root = Path(data_root)
    sequences = []

    if dataset == 'vimeo':
        gt_base = (data_root / 'target') if (data_root / 'target').exists() \
            else (data_root / 'sequences')

        # Each sequence: sr_dir/00001/0001/ with im1.png ~ im7.png
        for seq_dir in sorted(sr_dir.rglob('*')):
            if not seq_dir.is_dir():
                continue
            sr_frames = sorted(seq_dir.glob('im*.png'),
                               key=lambda p: int(p.stem[2:]))
            if len(sr_frames) < 2:
                continue
            rel = seq_dir.relative_to(sr_dir)
            gt_dir = gt_base / rel
            gt_frames = [gt_dir / f.name for f in sr_frames]
            if all(g.exists() for g in gt_frames):
                sequences.append({
                    'sr_frames': [str(f) for f in sr_frames],
                    'gt_frames': [str(f) for f in gt_frames],
                    'name': str(rel),
                })

    elif dataset in ('reds', 'reds4'):
        if dataset == 'reds':
            gt_base = data_root / 'val_sharp'
        else:
            gt_base = data_root / 'train_sharp'

        # Structure: sr_dir/<seq>/<center>/<frame>.png
        for seq_dir in sorted(sr_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            for sample_dir in sorted(seq_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sr_frames = sorted(sample_dir.glob('*.png'),
                                   key=lambda p: int(p.stem))
                if len(sr_frames) < 2:
                    continue
                gt_frames = [gt_base / seq_dir.name / f.name for f in sr_frames]
                if all(g.exists() for g in gt_frames):
                    sequences.append({
                        'sr_frames': [str(f) for f in sr_frames],
                        'gt_frames': [str(f) for f in gt_frames],
                        'name': f'{seq_dir.name}/{sample_dir.name}',
                    })

    return sequences


# ───────────────────── Visualization ─────────────────────

def create_comparison(sr_img, gt_img, lr_path=None, scale=4):
    """Side-by-side: [LR (bicubic upscaled)] | SR | GT with labels."""
    h, w = gt_img.shape[:2]
    panels, labels = [], []

    if lr_path and Path(lr_path).exists():
        lr_img = np.array(Image.open(lr_path).convert('RGB'))
        lr_up = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
        panels.append(lr_up)
        labels.append('LR (Bicubic)')

    panels.append(sr_img)
    labels.append('SR')
    panels.append(gt_img)
    labels.append('GT')

    labeled = []
    for panel, label in zip(panels, labels):
        p = panel.copy()
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(p, label, (10 + dx, 30 + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(p, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        labeled.append(p)

    return np.concatenate(labeled, axis=1)


# ───────────────────── Main ─────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate SR images against GT (shared across all methods)')
    p.add_argument('--sr_dir', type=str, required=True,
                   help='Directory containing SR images')
    p.add_argument('--dataset', type=str, required=True,
                   choices=['vimeo', 'reds', 'reds4'])
    p.add_argument('--data_root', type=str, required=True,
                   help='Path to dataset root (vimeo_septuplet/ or reds/)')
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--crop_border', type=int, default=0,
                   help='Crop N border pixels before computing PSNR/SSIM')
    # Metric flags
    p.add_argument('--lpips', action='store_true',
                   help='Compute LPIPS (requires: pip install lpips)')
    p.add_argument('--fid', action='store_true',
                   help='Compute FID (requires: pip install pytorch-fid)')
    p.add_argument('--tlpips', action='store_true',
                   help='Compute tLPIPS temporal consistency '
                        '(requires --save_all_frames during inference)')
    # Visualization
    p.add_argument('--visualize', type=int, default=0,
                   help='Generate N comparison images for qualitative analysis')
    p.add_argument('--vis_dir', type=str, default=None,
                   help='Directory for comparison images (default: <sr_dir>/visualizations)')
    p.add_argument('--device', type=str, default=None,
                   help='Device for LPIPS/FID (default: auto)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Find center-frame pairs ──
    pairs = find_center_pairs(args.sr_dir, args.dataset, args.data_root)
    if not pairs:
        print(f'ERROR: No SR/GT pairs found in {args.sr_dir}')
        print(f'  Expected SR images matching GT in {args.data_root}')
        return

    print(f'Found {len(pairs)} center-frame SR/GT pairs')

    # Init LPIPS if needed (shared by --lpips and --tlpips)
    lpips_metric = None
    if args.lpips or args.tlpips:
        try:
            lpips_metric = LPIPSMetric(device=args.device)
            print(f'LPIPS initialized (device={args.device})')
        except ImportError:
            print('WARNING: lpips not installed. Install with: pip install lpips')
            if args.lpips:
                args.lpips = False
            if args.tlpips:
                args.tlpips = False

    # ── 2. Per-image metrics: PSNR, SSIM, LPIPS ──
    psnr_list, ssim_list, lpips_list = [], [], []

    for pair in tqdm(pairs, desc='PSNR/SSIM' + ('/LPIPS' if args.lpips else '')):
        sr_img = np.array(Image.open(pair['sr']).convert('RGB'))
        gt_img = np.array(Image.open(pair['gt']).convert('RGB'))

        psnr_list.append(calc_psnr(sr_img, gt_img, crop_border=args.crop_border))
        ssim_list.append(calc_ssim(sr_img, gt_img, crop_border=args.crop_border))

        if args.lpips and lpips_metric:
            lpips_list.append(lpips_metric.calc(sr_img, gt_img))

    # ── 3. FID ──
    fid_value = None
    if args.fid:
        try:
            print('Computing FID...')
            fid_value = compute_fid(pairs, args.device)
        except ImportError:
            print('WARNING: pytorch-fid not installed. Install with: pip install pytorch-fid')
        except Exception as e:
            print(f'WARNING: FID computation failed: {e}')

    # ── 4. tLPIPS ──
    tlpips_value = None
    if args.tlpips and lpips_metric:
        sequences = find_sequences_for_tlpips(args.sr_dir, args.dataset, args.data_root)
        if not sequences:
            print('WARNING: No multi-frame sequences found for tLPIPS.')
            print('  Run inference with --save_all_frames first.')
        else:
            print(f'Computing tLPIPS on {len(sequences)} sequences...')
            tlpips_scores = []
            for seq in tqdm(sequences, desc='tLPIPS'):
                sr_frames = [np.array(Image.open(f).convert('RGB'))
                             for f in seq['sr_frames']]
                gt_frames = [np.array(Image.open(f).convert('RGB'))
                             for f in seq['gt_frames']]
                tlpips_scores.append(lpips_metric.calc_tlpips(sr_frames, gt_frames))
            tlpips_value = np.mean(tlpips_scores)

    # ── 5. Print results ──
    sep = '=' * 55
    print(f'\n{sep}')
    print(f'  Dataset : {args.dataset}')
    print(f'  SR dir  : {args.sr_dir}')
    print(f'  Samples : {len(pairs)}')
    print(f'{sep}')
    print(f'  PSNR    : {np.mean(psnr_list):.4f} dB')
    print(f'  SSIM    : {np.mean(ssim_list):.6f}')
    if lpips_list:
        print(f'  LPIPS   : {np.mean(lpips_list):.6f}')
    if fid_value is not None:
        print(f'  FID     : {fid_value:.4f}')
    if tlpips_value is not None:
        print(f'  tLPIPS  : {tlpips_value:.6f}')
    print(f'{sep}')

    # ── 6. Save results ──
    sr_dir = Path(args.sr_dir)

    # Per-image CSV
    csv_path = sr_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['name', 'psnr', 'ssim']
        if lpips_list:
            header.append('lpips')
        writer.writerow(header)
        for i, pair in enumerate(pairs):
            row = [pair['name'], f'{psnr_list[i]:.4f}', f'{ssim_list[i]:.6f}']
            if lpips_list:
                row.append(f'{lpips_list[i]:.6f}')
            writer.writerow(row)

    # Summary
    summary_path = sr_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'SR dir: {args.sr_dir}\n')
        f.write(f'Samples: {len(pairs)}\n')
        f.write(f'PSNR: {np.mean(psnr_list):.4f}\n')
        f.write(f'SSIM: {np.mean(ssim_list):.6f}\n')
        if lpips_list:
            f.write(f'LPIPS: {np.mean(lpips_list):.6f}\n')
        if fid_value is not None:
            f.write(f'FID: {fid_value:.4f}\n')
        if tlpips_value is not None:
            f.write(f'tLPIPS: {tlpips_value:.6f}\n')

    print(f'Per-image metrics: {csv_path}')
    print(f'Summary: {summary_path}')

    # ── 7. Qualitative visualization ──
    if args.visualize > 0:
        vis_dir = Path(args.vis_dir) if args.vis_dir else sr_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        n = min(args.visualize, len(pairs))
        indices = np.linspace(0, len(pairs) - 1, n, dtype=int)

        for i, idx in enumerate(indices):
            pair = pairs[idx]
            sr_img = np.array(Image.open(pair['sr']).convert('RGB'))
            gt_img = np.array(Image.open(pair['gt']).convert('RGB'))

            comp = create_comparison(sr_img, gt_img, pair.get('lr'), args.scale)
            safe_name = pair['name'].replace('/', '_').replace('\\', '_')
            save_path = vis_dir / f'compare_{i:03d}_{safe_name}'
            Image.fromarray(comp).save(save_path)

        print(f'Saved {n} comparison images to: {vis_dir}')


if __name__ == '__main__':
    main()
