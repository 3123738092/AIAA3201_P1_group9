# Part 1: Temporal Baseline — Hand-crafted Video Super-Resolution

Implementation of Section 5.1 from the AIAA3201 Project 1 spec.

## Experiments Overview

We conduct **7 experiments** in total, progressively combining spatial upsampling, temporal fusion, and sharpening:

### Group A: Spatial-Only (Single Frame)

These methods process each frame independently, no temporal information is used.

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask |
|---|-----------|---------------|----------------|-------------|
| 1 | Bicubic | Bicubic | - | - |
| 2 | Lanczos | Lanczos | - | - |
| 3 | SRCNN | Bicubic + SRCNN | - | - |

### Group B: Temporal Fusion (Multi-Frame, Bicubic-based)

First bicubic-upsample each frame, then fuse neighboring frames via Gaussian-weighted averaging (window=5).

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask |
|---|-----------|---------------|----------------|-------------|
| 4 | Bicubic + Temporal Avg | Bicubic | Gaussian w=5 | - |
| 5 | Bicubic + Temporal Avg + USM | Bicubic | Gaussian w=5 | sigma=1.0, strength=0.5 |

### Group C: Temporal Fusion (Multi-Frame, SRCNN-based)

First bicubic-upsample + SRCNN refine each frame, then fuse neighboring frames.

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask |
|---|-----------|---------------|----------------|-------------|
| 6 | SRCNN + Temporal Avg | Bicubic + SRCNN | Gaussian w=5 | - |
| 7 | SRCNN + Temporal Avg + USM | Bicubic + SRCNN | Gaussian w=5 | sigma=1.0, strength=0.5 |

### Pipeline Diagram

```
LR Frame t-2 ─┐
LR Frame t-1 ─┤
LR Frame t   ─┼─→ [Spatial Upsample] ─→ [Temporal Weighted Avg] ─→ [Unsharp Mask] ─→ SR Frame t
LR Frame t+1 ─┤    (Bicubic/SRCNN)       (Gaussian, window=5)       (optional)
LR Frame t+2 ─┘
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Supported datasets: **Vimeo-90K Septuplet** and **REDS**.

```
AIAA3201_P1_group9/
├── data/
│   ├── vimeo_septuplet/       # or vimeo_super_resolution_test/
│   │   ├── sequences/         # (septuplet) or target/ + low_resolution/ (sr_test)
│   │   ├── sep_trainlist.txt
│   │   └── sep_testlist.txt
│   └── reds/
│       ├── train_sharp/          # GT: 000~239, 100 frames each
│       ├── train_sharp_bicubic/X4/
│       ├── val_sharp/            # GT: 000~029
│       └── val_sharp_bicubic/X4/
└── temporal_baseline/
```

## Workflow

The workflow is split into **two decoupled stages**:

1. **`inference.py`** — generates SR images and saves them to disk
2. **`evaluate.py`** (repo root) — reads SR images, computes metrics, creates visualizations

This design allows `evaluate.py` to be **shared across all methods** (temporal baseline, Real-ESRGAN, BasicVSR, etc.) — any method that outputs SR images in the same format can be evaluated identically.

### Step 1: Train SRCNN

```bash
cd temporal_baseline

# Train on Vimeo-90K
python train_srcnn.py --dataset vimeo --data_root /path/to/vimeo_septuplet --device cuda

# Train on REDS
python train_srcnn.py --dataset reds --data_root /path/to/reds --device cuda
```

### Step 2: Generate SR Images

Each command generates SR center-frame images to `results/<dataset>/<method>/`.

```bash
cd temporal_baseline

# --- Group A: Spatial-Only ---
python inference.py --dataset vimeo --method bicubic --data_root /path/to/vimeo_septuplet
python inference.py --dataset vimeo --method lanczos --data_root /path/to/vimeo_septuplet
python inference.py --dataset vimeo --method srcnn   --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

# --- Group B: Bicubic + Temporal ---
python inference.py --dataset vimeo --method temporal_avg --base_method bicubic --window 5 --data_root /path/to/vimeo_septuplet
python inference.py --dataset vimeo --method temporal_avg --base_method bicubic --window 5 --unsharp --data_root /path/to/vimeo_septuplet

# --- Group C: SRCNN + Temporal ---
python inference.py --dataset vimeo --method temporal_avg --base_method srcnn --window 5 --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth
python inference.py --dataset vimeo --method temporal_avg --base_method srcnn --window 5 --unsharp --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

# Quick test (limit to 10 samples)
python inference.py --dataset vimeo --method bicubic --data_root /path/to/vimeo_septuplet --max_sequences 10
```

### Step 3: Evaluate (Quantitative)

```bash
cd AIAA3201_P1_group9  # repo root

# Basic: PSNR + SSIM (always computed)
python evaluate.py --sr_dir results/vimeo/bicubic --dataset vimeo --data_root data/vimeo_septuplet

# All mandatory metrics: PSNR + SSIM + LPIPS + FID
python evaluate.py --sr_dir results/vimeo/srcnn --dataset vimeo --data_root data/vimeo_septuplet --lpips --fid

# tLPIPS requires --save_all_frames during inference:
python evaluate.py --sr_dir results/vimeo/bicubic --dataset vimeo --data_root data/vimeo_septuplet --lpips --fid --tlpips
```

Output per method:
- `metrics.csv` — per-image PSNR, SSIM (, LPIPS)
- `metrics_summary.txt` — averaged scores (all metrics)

### Step 4: Visualize (Qualitative)

```bash
# Generate 10 side-by-side comparison images (LR | SR | GT)
python evaluate.py --sr_dir results/vimeo/bicubic --dataset vimeo --data_root data/vimeo_septuplet --visualize 10

# Custom output directory
python evaluate.py --sr_dir results/vimeo/srcnn --dataset vimeo --data_root data/vimeo_septuplet --visualize 5 --vis_dir results/figures/srcnn_vimeo
```

### Run All 7 Experiments at Once

```bash
cd temporal_baseline

# Basic: PSNR + SSIM only (fastest)
python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

# All mandatory metrics + visualizations
python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth \
    --lpips --fid --tlpips --visualize 5

# REDS
python evaluate_all.py --dataset reds --data_root /path/to/reds --checkpoint checkpoints/srcnn_best.pth --lpips --fid --tlpips

# Quick test (10 samples)
python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth --max_sequences 10
```

## Output Structure

```
results/
├── vimeo/
│   ├── bicubic/
│   │   ├── 00001/0001/im4.png      # SR center frame
│   │   ├── 00002/0003/im4.png
│   │   ├── metrics.csv              # per-image scores
│   │   ├── metrics_summary.txt      # averaged scores
│   │   └── visualizations/          # comparison images (if --visualize)
│   │       ├── compare_000_00001_0001_im4.png
│   │       └── ...
│   ├── lanczos/
│   ├── srcnn/
│   ├── temporal_avg_bicubic_w5/
│   ├── temporal_avg_bicubic_w5_usm/
│   ├── temporal_avg_srcnn_w5/
│   └── temporal_avg_srcnn_w5_usm/
└── reds/
    └── ... (same structure, images named as 000/00000003.png)
```

## Metrics (matching Project Guideline §6.2)

All mandatory metrics from AIAA3201 Project 1:

| Metric | Category | Flag | Direction | Package |
|--------|----------|------|-----------|---------|
| **PSNR** | Pixel Accuracy | *(always)* | higher ↑ | scikit-image |
| **SSIM** | Pixel Accuracy | *(always)* | higher ↑ | scikit-image |
| **LPIPS** | Perceptual Quality | `--lpips` | lower ↓ | `pip install lpips` |
| **FID** | Perceptual Quality | `--fid` | lower ↓ | `pip install pytorch-fid` |
| **tLPIPS** | Temporal Consistency | `--tlpips` | lower ↓ | lpips (+ `--save_all_frames`) |

> **Note:** tLPIPS measures flickering by comparing frame-difference patterns.
> It requires saving all frames during inference (`--save_all_frames`).
> When using `evaluate_all.py --tlpips`, this flag is auto-enabled.

## Expected Results (to be filled after evaluation)

| # | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | tLPIPS ↓ |
|---|--------|--------|--------|---------|-------|----------|
| 1 | Bicubic | - | - | - | - | - |
| 2 | Lanczos | - | - | - | - | - |
| 3 | SRCNN | - | - | - | - | - |
| 4 | Bicubic + Temporal Avg | - | - | - | - | - |
| 5 | Bicubic + Temporal Avg + USM | - | - | - | - | - |
| 6 | SRCNN + Temporal Avg | - | - | - | - | - |
| 7 | SRCNN + Temporal Avg + USM | - | - | - | - | - |
