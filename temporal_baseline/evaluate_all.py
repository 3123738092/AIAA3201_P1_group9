"""Run all Part 1 methods and produce a comparison table.

Usage:
    python evaluate_all.py --data_root ../../data/vimeo_septuplet --checkpoint checkpoints/srcnn_best.pth
"""

import argparse
import os
import sys
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='checkpoints/srcnn_best.pth')
    p.add_argument('--max_sequences', type=int, default=0)
    p.add_argument('--scale', type=int, default=4)
    args = p.parse_args()

    base_cmd = [sys.executable, 'inference.py', '--data_root', args.data_root, '--scale', str(args.scale)]
    if args.max_sequences > 0:
        base_cmd += ['--max_sequences', str(args.max_sequences)]

    methods = [
        # (description, extra_args)
        ('Bicubic', ['--method', 'bicubic']),
        ('Lanczos', ['--method', 'lanczos']),
        ('SRCNN', ['--method', 'srcnn', '--checkpoint', args.checkpoint]),
        ('Temporal Avg (Bicubic, w=5)', ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5']),
        ('Temporal Avg (Bicubic, w=5) + USM', ['--method', 'temporal_avg', '--base_method', 'bicubic', '--window', '5', '--unsharp']),
        ('Temporal Avg (SRCNN, w=5)', ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5']),
        ('Temporal Avg (SRCNN, w=5) + USM', ['--method', 'temporal_avg', '--base_method', 'srcnn', '--checkpoint', args.checkpoint, '--window', '5', '--unsharp']),
    ]

    print('=' * 60)
    print('Part 1 Baseline Evaluation on Vimeo-90K')
    print('=' * 60)

    for desc, extra in methods:
        print(f'\n>>> {desc}')
        cmd = base_cmd + extra
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    main()
