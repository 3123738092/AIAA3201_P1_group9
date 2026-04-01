# Part 1: Temporal Baseline — Hand-crafted Video Super-Resolution

Implementation of Section 5.1 from the AIAA3201 Project 1 spec.

## Experiments Overview

We conduct **7 experiments** in total, progressively combining spatial upsampling, temporal fusion, and sharpening:

### Group A: Spatial-Only (Single Frame)

These methods process each frame independently, no temporal information is used.

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask | Command |
|---|-----------|---------------|----------------|-------------|---------|
| 1 | Bicubic | Bicubic | - | - | `--method bicubic` |
| 2 | Lanczos | Lanczos | - | - | `--method lanczos` |
| 3 | SRCNN | Bicubic + SRCNN | - | - | `--method srcnn` |

### Group B: Temporal Fusion (Multi-Frame, Bicubic-based)

First bicubic-upsample each frame, then fuse neighboring frames via Gaussian-weighted averaging (window=5).

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask | Command |
|---|-----------|---------------|----------------|-------------|---------|
| 4 | Bicubic + Temporal Avg | Bicubic | Gaussian w=5 | - | `--method temporal_avg --base_method bicubic --window 5` |
| 5 | Bicubic + Temporal Avg + USM | Bicubic | Gaussian w=5 | sigma=1.0, strength=0.5 | `--method temporal_avg --base_method bicubic --window 5 --unsharp` |

### Group C: Temporal Fusion (Multi-Frame, SRCNN-based)

First bicubic-upsample + SRCNN refine each frame, then fuse neighboring frames.

| # | Experiment | Spatial Method | Temporal Fusion | Unsharp Mask | Command |
|---|-----------|---------------|----------------|-------------|---------|
| 6 | SRCNN + Temporal Avg | Bicubic + SRCNN | Gaussian w=5 | - | `--method temporal_avg --base_method srcnn --window 5` |
| 7 | SRCNN + Temporal Avg + USM | Bicubic + SRCNN | Gaussian w=5 | sigma=1.0, strength=0.5 | `--method temporal_avg --base_method srcnn --window 5 --unsharp` |

### Pipeline Diagram

```
LR Frame t-2 ─┐
LR Frame t-1 ─┤
LR Frame t   ─┼─→ [Spatial Upsample] ─→ [Temporal Weighted Avg] ─→ [Unsharp Mask] ─→ SR Frame t
LR Frame t+1 ─┤    (Bicubic/SRCNN)       (Gaussian, window=5)       (optional)
LR Frame t+2 ─┘
```

### Expected Results (to be filled after evaluation)

| # | Method | PSNR (dB) | SSIM | Notes |
|---|--------|-----------|------|-------|
| 1 | Bicubic | - | - | Absolute baseline |
| 2 | Lanczos | - | - | Slightly sharper than bicubic |
| 3 | SRCNN | - | - | Learned mapping, slight improvement |
| 4 | Bicubic + Temporal Avg | - | - | Multi-frame denoising, may improve PSNR |
| 5 | Bicubic + Temporal Avg + USM | - | - | Sharpened edges on top of averaging |
| 6 | SRCNN + Temporal Avg | - | - | Best CNN + temporal fusion |
| 7 | SRCNN + Temporal Avg + USM | - | - | Full pipeline |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download **Vimeo-90K Septuplet** from http://toflow.csail.mit.edu/ and place under:

```
AIAA3201_P1_group9/
└── data/
    └── vimeo_septuplet/
        ├── sequences/
        ├── sep_trainlist.txt
        └── sep_testlist.txt
```

**Why Septuplet?** The temporal baseline uses multi-frame weighted averaging (window=5). Septuplet provides 7 frames per sequence, giving sufficient temporal context for the center frame.

## Usage

### 1. Train SRCNN

```bash
cd temporal_baseline
python train_srcnn.py --data_root ../data/vimeo_septuplet --epochs 20 --batch_size 16
```

### 2. Run all 7 experiments at once

```bash
python evaluate_all.py --data_root ../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth
```

### 3. Or run individual experiments

```bash
# Exp 1: Bicubic
python inference.py --method bicubic --data_root ../data/vimeo_septuplet

# Exp 2: Lanczos
python inference.py --method lanczos --data_root ../data/vimeo_septuplet

# Exp 3: SRCNN
python inference.py --method srcnn --data_root ../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

# Exp 4: Bicubic + Temporal Avg
python inference.py --method temporal_avg --data_root ../data/vimeo_septuplet --base_method bicubic --window 5

# Exp 5: Bicubic + Temporal Avg + Unsharp Mask
python inference.py --method temporal_avg --data_root ../data/vimeo_septuplet --base_method bicubic --window 5 --unsharp

# Exp 6: SRCNN + Temporal Avg
python inference.py --method temporal_avg --data_root ../data/vimeo_septuplet --base_method srcnn --checkpoint checkpoints/srcnn_best.pth --window 5

# Exp 7: SRCNN + Temporal Avg + Unsharp Mask
python inference.py --method temporal_avg --data_root ../data/vimeo_septuplet --base_method srcnn --checkpoint checkpoints/srcnn_best.pth --window 5 --unsharp
```

Add `--save_images` to save output frames. Add `--max_sequences 10` for a quick test.

Results are saved to `results/temporal_baseline/`.

## Metrics

- **PSNR** (dB) — pixel-level accuracy
- **SSIM** — structural similarity
- **LPIPS** — perceptual quality (available via `utils/metrics.py`)
- **tLPIPS** — temporal consistency between consecutive frames
