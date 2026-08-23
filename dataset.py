import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from main.normalize import compute_log_params, build_input_tensor
#synthetic data examples from my renders . old.
DEPTH_GLOB = "data/frame_*_depth_proj_mm.npy"
MASK_TMPL = "data/frame_{idx}_proj_valid_mask.npy"
RGB_TMPL = "data/frame_{idx}_rgb.png"  # adjust ext if not png
GT_DEPTH_TMPL = "data/frame_{idx}_gt_mm.npy"
GT_MASK_TMPL = "data/frame_{idx}_gt_mask.npy"


def _extract_idx(depth_path):
    m = re.search(r"frame_(\d+)_depth_proj_mm\.npy", depth_path)
    return m.group(1)


class DToFDataset(Dataset):
    def __init__(self, root="."):
        self.root = root
        self.depth_files = sorted(glob.glob(os.path.join(root, DEPTH_GLOB)))
        self.alpha, self.beta = compute_log_params()

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, i):
        depth_path = self.depth_files[i]
        idx = _extract_idx(depth_path)

        mask_path = os.path.join(self.root, MASK_TMPL.format(idx=idx))
        rgb_path = os.path.join(self.root, RGB_TMPL.format(idx=idx))
        gt_depth_path = os.path.join(self.root, GT_DEPTH_TMPL.format(idx=idx))
        gt_mask_path = os.path.join(self.root, GT_MASK_TMPL.format(idx=idx))

        depth = np.load(depth_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        gt_depth = np.load(gt_depth_path).astype(np.float32)
        gt_mask = np.load(gt_mask_path).astype(np.float32)

        depth_tensor = build_input_tensor(depth, mask, self.alpha, self.beta)

        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),          # (3,H,W)
            "depth_input": torch.from_numpy(depth_tensor),          # (3,H,W) in [-1,1]
            "valid_mask": torch.from_numpy(mask),                   # (H,W)
            "gt_depth": torch.from_numpy(gt_depth).unsqueeze(0),    # (1,H,W) metric mm
            "gt_mask": torch.from_numpy(gt_mask).unsqueeze(0),      # (1,H,W)
            "alpha": self.alpha,
            "beta": self.beta,
            "frame_idx": idx,
        }


if __name__ == "__main__":
    ds = DToFDataset()
    print(len(ds), "frames")
    sample = ds[0]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
