# Part 1: Temporal Baseline — Hand-crafted Video Super-Resolution

Implementation of Section 5.1 from the AIAA3201 Project 1 spec.

## Methods

| Method | Description |
|--------|-------------|
| Bicubic | Classic bicubic interpolation (baseline) |
| Lanczos | Lanczos resampling |
| SRCNN | 3-layer CNN refinement on bicubic-upscaled frames [Dong et al., TPAMI 2015] |
| Temporal Avg | Multi-frame Gaussian-weighted averaging across neighboring frames |
| + Unsharp Mask | Sharpening via USM after temporal averaging |

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

## Usage

### 1. Train SRCNN

```bash
cd temporal_baseline
python train_srcnn.py --data_root ../data/vimeo_septuplet --epochs 50 --batch_size 16
```

### 2. Run inference & evaluation

```bash
# Single method
python inference.py --method bicubic --data_root ../data/vimeo_septuplet
python inference.py --method lanczos --data_root ../data/vimeo_septuplet
python inference.py --method srcnn --data_root ../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth
python inference.py --method temporal_avg --data_root ../data/vimeo_septuplet --base_method bicubic --window 5 --unsharp

# All methods at once
python evaluate_all.py --data_root ../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth
```

Add `--save_images` to save output frames. Add `--max_sequences 10` for a quick test.

Results are saved to `results/temporal_baseline/`.

## Metrics

- **PSNR** (dB) — pixel-level accuracy
- **SSIM** — structural similarity
- **LPIPS** — perceptual quality (available via `utils/metrics.py`)
- **tLPIPS** — temporal consistency
