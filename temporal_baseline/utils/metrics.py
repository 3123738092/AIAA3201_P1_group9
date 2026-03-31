"""Evaluation metrics: PSNR, SSIM, LPIPS, tLPIPS."""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calc_psnr(img1, img2, crop_border=0):
    """Calculate PSNR between two images (numpy HWC uint8 or float64 [0,1])."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def calc_ssim(img1, img2, crop_border=0):
    """Calculate SSIM between two images (numpy HWC uint8 or float64 [0,1])."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    return structural_similarity(img1, img2, multichannel=True, channel_axis=-1, data_range=1.0)


class LPIPSMetric:
    """Wrapper for LPIPS perceptual metric."""

    def __init__(self, net='alex', device='cuda'):
        import lpips
        self.device = device
        self.fn = lpips.LPIPS(net=net).to(device).eval()

    def _to_tensor(self, img):
        """numpy HWC [0,1] float -> tensor BCHW [-1,1]."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return (t * 2.0 - 1.0).to(self.device)

    @torch.no_grad()
    def calc_lpips(self, img1, img2):
        return self.fn(self._to_tensor(img1), self._to_tensor(img2)).item()

    @torch.no_grad()
    def calc_tlpips(self, frames_sr, frames_gt):
        """Temporal LPIPS: average LPIPS between consecutive frame differences.

        tLPIPS = mean_t LPIPS( SR_t - SR_{t-1}, GT_t - GT_{t-1} )
        """
        scores = []
        for i in range(1, len(frames_sr)):
            diff_sr = frames_sr[i].astype(np.float32) - frames_sr[i - 1].astype(np.float32)
            diff_gt = frames_gt[i].astype(np.float32) - frames_gt[i - 1].astype(np.float32)
            # normalize diffs to [0,1] for LPIPS
            diff_sr = np.clip(diff_sr / 2.0 + 0.5, 0, 1)
            diff_gt = np.clip(diff_gt / 2.0 + 0.5, 0, 1)
            scores.append(self.calc_lpips(diff_sr, diff_gt))
        return np.mean(scores) if scores else 0.0
