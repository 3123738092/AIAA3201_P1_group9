"""Train SRCNN on Vimeo-90K (single-frame super-resolution).

Usage:
    python train_srcnn.py --data_root ../../data/vimeo_septuplet --epochs 50 --batch_size 16
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets.vimeo90k import Vimeo90KSeptuplet
from models.srcnn import SRCNN
from utils.metrics import calc_psnr, calc_ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True, help='Path to vimeo_septuplet/')
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--patch_size', type=int, default=256, help='GT patch size for training')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scale):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        lr = batch['lr'].to(device)   # (B, 7, 3, h, w)
        gt = batch['gt'].to(device)   # (B, 7, 3, H, W)
        B, T, C, H, W = gt.shape

        # Train on each frame independently; bicubic upsample LR first
        lr_up = torch.nn.functional.interpolate(
            lr.view(B * T, C, H // scale, W // scale), size=(H, W), mode='bicubic', align_corners=False
        )
        gt_flat = gt.view(B * T, C, H, W)

        pred = model(lr_up)
        loss = criterion(pred, gt_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, scale):
    model.eval()
    psnr_sum, ssim_sum, count = 0, 0, 0
    for batch in tqdm(loader, desc='Val', leave=False):
        lr = batch['lr'].to(device)
        gt = batch['gt'].to(device)
        B, T, C, H, W = gt.shape

        lr_up = torch.nn.functional.interpolate(
            lr.view(B * T, C, H // scale, W // scale), size=(H, W), mode='bicubic', align_corners=False
        )
        pred = model(lr_up).clamp(0, 1)
        gt_flat = gt.view(B * T, C, H, W)

        # Evaluate center frame only (frame index 3 for septuplet)
        for b in range(B):
            center = 3  # middle of 7 frames
            p = pred[b * T + center].cpu().numpy().transpose(1, 2, 0)
            g = gt_flat[b * T + center].cpu().numpy().transpose(1, 2, 0)
            psnr_sum += calc_psnr(p, g)
            ssim_sum += calc_ssim(p, g)
            count += 1

    return psnr_sum / count, ssim_sum / count


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    train_set = Vimeo90KSeptuplet(args.data_root, split='train', scale=args.scale, patch_size=args.patch_size)
    test_set = Vimeo90KSeptuplet(args.data_root, split='test', scale=args.scale)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    model = SRCNN().to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device, args.scale)
        psnr, ssim = validate(model, test_loader, args.device, args.scale)
        scheduler.step()

        print(f'Epoch {epoch}/{args.epochs}  Loss: {loss:.6f}  PSNR: {psnr:.2f}  SSIM: {ssim:.4f}')

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'srcnn_best.pth'))
            print(f'  -> Saved best model (PSNR={psnr:.2f})')

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'srcnn_last.pth'))
    print(f'Training done. Best PSNR: {best_psnr:.2f}')


if __name__ == '__main__':
    main()
