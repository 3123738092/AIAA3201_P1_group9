"""Run all Part 1 methods: generate SR images, then evaluate each.

Usage:
    # Vimeo-90K (full pipeline)
    python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet \
                           --checkpoint checkpoints/srcnn_best.pth

    # REDS val
    python evaluate_all.py --dataset reds --data_root /path/to/reds \
                           --checkpoint checkpoints/srcnn_best.pth

    # Quick test (10 samples only)
    python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet \
                           --checkpoint checkpoints/srcnn_best.pth --max_sequences 10

    # With qualitative visualizations
    python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet \
                           --checkpoint checkpoints/srcnn_best.pth --visualize 5
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='reds', choices=['vimeo', 'reds', 'reds4'])
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='checkpoints/srcnn_best.pth')
    p.add_argument('--output_dir', type=str, default='results',
                   help='Base output directory for SR images')
    p.add_argument('--max_sequences', type=int, default=0)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--visualize', type=int, default=0,
                   help='Generate N comparison images per method')
    p.add_argument('--lpips', action='store_true', help='Compute LPIPS metric')
    p.add_argument('--fid', action='store_true', help='Compute FID metric')
    p.add_argument('--tlpips', action='store_true',
                   help='Compute tLPIPS (computed in-memory during inference, no extra disk)')
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    evaluate_script = repo_root / 'evaluate.py'

    # Base command for inference
    infer_base = [
        sys.executable, str(script_dir / 'inference.py'),
        '--dataset', args.dataset,
        '--data_root', args.data_root,
        '--scale', str(args.scale),
        '--output_dir', args.output_dir,
    ]
    if args.max_sequences > 0:
        infer_base += ['--max_sequences', str(args.max_sequences)]
    if args.tlpips:
        infer_base += ['--tlpips']

    # Base command for evaluation
    eval_base = [
        sys.executable, str(evaluate_script),
        '--dataset', args.dataset,
        '--data_root', args.data_root,
    ]
    if args.visualize > 0:
        eval_base += ['--visualize', str(args.visualize)]
    if args.lpips:
        eval_base += ['--lpips']
    if args.fid:
        eval_base += ['--fid']

    # 7 experiments
    methods = [
        ('Exp1: Bicubic',
         ['--method', 'bicubic'],
         'bicubic'),
        ('Exp2: Lanczos',
         ['--method', 'lanczos'],
         'lanczos'),
        ('Exp3: SRCNN',
         ['--method', 'srcnn', '--checkpoint', args.checkpoint],
         'srcnn'),
        ('Exp4: Bicubic + Temporal Avg (w=5)',
         ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5'],
         'temporal_avg_bicubic_w5'),
        ('Exp5: Bicubic + Temporal Avg (w=5) + USM',
         ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5', '--unsharp'],
         'temporal_avg_bicubic_w5_usm'),
        ('Exp6: SRCNN + Temporal Avg (w=5)',
         ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5'],
         'temporal_avg_srcnn_w5'),
        ('Exp7: SRCNN + Temporal Avg (w=5) + USM',
         ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5', '--unsharp'],
         'temporal_avg_srcnn_w5_usm'),
    ]

    print('=' * 60)
    print(f'Part 1 Baseline: Generate SR + Evaluate — {args.dataset.upper()}')
    print('=' * 60)

    for desc, infer_extra, method_name in methods:
        print(f'\n{"─"*60}')
        print(f'>>> {desc}')
        print(f'{"─"*60}')

        # Step 1: Generate SR images
        infer_cmd = infer_base + infer_extra
        subprocess.run(infer_cmd)

        # Step 2: Evaluate
        sr_dir = str(Path(args.output_dir) / args.dataset / method_name)
        eval_cmd = eval_base + ['--sr_dir', sr_dir]
        subprocess.run(eval_cmd)

    print(f'\n{"="*60}')
    print('All experiments complete!')
    print(f'Results saved to: {args.output_dir}/{args.dataset}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
