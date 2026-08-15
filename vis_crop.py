#!/usr/bin/env python3
"""Visualize center-cropped frame_0000 data at 224x224."""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def load_frame_0000():
    rgb = np.array(Image.open('data/frame_0008_rgb.png')).astype(np.float32) / 255.0
    depth = np.load('data/frame_0008_depth_proj_mm.npy').astype(np.float32)
    mask = np.load('data/frame_0008_proj_valid_mask.npy').astype(np.float32)
    gt_depth = np.load('data/frame_0008_gt_mm.npy').astype(np.float32)
    gt_mask = np.load('data/frame_0008_gt_mask.npy').astype(np.float32)
    return rgb, depth, mask, gt_depth, gt_mask

def center_crop(arr, target_h, target_w):
    """Crop center region of array."""
    h, w = arr.shape[:2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    if arr.ndim == 3:
        return arr[top:top+target_h, left:left+target_w, :]
    else:
        return arr[top:top+target_h, left:left+target_w]

def save_vis(name, arr, out_dir, cmap='turbo', vmin=None, vmax=None):
    """Save array as colored PNG."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        plt.imsave(os.path.join(out_dir, f"{name}.png"), np.clip(arr, 0, 1))
    else:
        valid = arr > 0
        if vmin is None:
            vmin = arr[valid].min() if valid.any() else 0
        if vmax is None:
            vmax = arr[valid].max() if valid.any() else 1
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
        colored = cm.get_cmap(cmap)(norm)[..., :3]
        plt.imsave(os.path.join(out_dir, f"{name}.png"), colored)
    print(f"  Saved {name}.png (shape={arr.shape}, range={arr.min():.1f}-{arr.max():.1f})")

def main():
    os.makedirs("crop_vis", exist_ok=True)

    print("Loading original 540x960...")
    rgb, depth, mask, gt_depth, gt_mask = load_frame_0000()
    print(f"  rgb: {rgb.shape}, depth: {depth.shape}, mask: {mask.shape}")

    # Center crop to 224x224
    print("\nCenter cropping to 224x224...")
    rgb_cc = center_crop(rgb, 224, 224)
    depth_cc = center_crop(depth, 224, 224)
    mask_cc = center_crop(mask, 224, 224)
    gt_depth_cc = center_crop(gt_depth, 224, 224)
    gt_mask_cc = center_crop(gt_mask, 224, 224)

    # Save cropped
    save_vis("cc224_rgb", rgb_cc, "crop_vis")
    save_vis("cc224_depth", depth_cc, "crop_vis")
    save_vis("cc224_mask", mask_cc, "crop_vis", cmap='gray')
    save_vis("cc224_gt_depth", gt_depth_cc, "crop_vis")
    save_vis("cc224_gt_mask", gt_mask_cc, "crop_vis", cmap='gray')

    # Stats
    print(f"\nOriginal depth valid: {mask.sum():.0f} pixels")
    print(f"Cropped depth valid: {mask_cc.sum():.0f} pixels")
    print(f"Original GT valid: {gt_mask.sum():.0f} pixels")
    print(f"Cropped GT valid: {gt_mask_cc.sum():.0f} pixels")

    print("\nDone! Check crop_vis/")

if __name__ == "__main__":
    main()
