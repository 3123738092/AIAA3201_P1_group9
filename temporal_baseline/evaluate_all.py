"""Run all Part 1 methods and produce a comparison table.

Usage:
    # REDS
    python evaluate_all.py --dataset reds --data_root /path/to/reds --checkpoint checkpoints/srcnn_best.pth

    # Vimeo-90K
    python evaluate_all.py --dataset vimeo --data_root /path/to/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth

    # REDS4 (standard benchmark)
    python evaluate_all.py --dataset reds4 --data_root /path/to/reds --checkpoint checkpoints/srcnn_best.pth
"""

import argparse
import os
import sys
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='reds', choices=['vimeo', 'reds', 'reds4'])
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='checkpoints/srcnn_best.pth')
    p.add_argument('--max_sequences', type=int, default=0)
    p.add_argument('--scale', type=int, default=4)
    args = p.parse_args()

    base_cmd = [
        sys.executable, 'inference.py',
        '--dataset', args.dataset,
        '--data_root', args.data_root,
        '--scale', str(args.scale),
    ]
    if args.max_sequences > 0:
        base_cmd += ['--max_sequences', str(args.max_sequences)]

    methods = [
        ('Exp1: Bicubic',
         ['--method', 'bicubic']),
        ('Exp2: Lanczos',
         ['--method', 'lanczos']),
        ('Exp3: SRCNN',
         ['--method', 'srcnn', '--checkpoint', args.checkpoint]),
        ('Exp4: Bicubic + Temporal Avg (w=5)',
         ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5']),
        ('Exp5: Bicubic + Temporal Avg (w=5) + USM',
         ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5', '--unsharp']),
        ('Exp6: SRCNN + Temporal Avg (w=5)',
         ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5']),
        ('Exp7: SRCNN + Temporal Avg (w=5) + USM',
         ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5', '--unsharp']),
    ]

    print('=' * 60)
    print(f'Part 1 Baseline Evaluation — {args.dataset.upper()}')
    print('=' * 60)

    for desc, extra in methods:
        print(f'\n>>> {desc}')
        cmd = base_cmd + extra
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    main()
