#!/usr/bin/env python3
"""Visualize downsampled frame_0000 data at 224x224."""
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def load_frame_0000():
    rgb = np.array(Image.open('data/frame_0000_rgb.png')).astype(np.float32) / 255.0
    depth = np.load('data/frame_0000_depth_proj_mm.npy').astype(np.float32)
    mask = np.load('data/frame_0000_proj_valid_mask.npy').astype(np.float32)
    gt_depth = np.load('data/frame_0000_gt_mm.npy').astype(np.float32)
    gt_mask = np.load('data/frame_0000_gt_mask.npy').astype(np.float32)
    return rgb, depth, mask, gt_depth, gt_mask

def downsample_data(rgb, depth, mask, gt_depth, gt_mask, target_h, target_w):
    rgb_ds = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    depth_ds = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask_ds = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    gt_mask_ds = cv2.resize(gt_mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    gt_depth_ds = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return rgb_ds, depth_ds, mask_ds, gt_depth_ds, gt_mask_ds

def save_vis(name, arr, out_dir, cmap='turbo', vmin=None, vmax=None):
    """Save array as colored PNG."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        # RGB
        plt.imsave(os.path.join(out_dir, f"{name}.png"), np.clip(arr, 0, 1))
    else:
        # Depth/mask - colorize
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
    os.makedirs("downsample_vis", exist_ok=True)

    print("Loading original 540x960...")
    rgb, depth, mask, gt_depth, gt_mask = load_frame_0000()
    print(f"  rgb: {rgb.shape}, depth: {depth.shape}, mask: {mask.shape}")
    print(f"  gt_depth: {gt_depth.shape}, gt_mask: {gt_mask.shape}")

    # Save original
    save_vis("orig_rgb", rgb, "downsample_vis")
    save_vis("orig_depth", depth, "downsample_vis")
    save_vis("orig_mask", mask, "downsample_vis", cmap='gray')
    save_vis("orig_gt_depth", gt_depth, "downsample_vis")
    save_vis("orig_gt_mask", gt_mask, "downsample_vis", cmap='gray')

    # Downsample to 224x224
    print("\nDownsampling to 224x224...")
    rgb_lr, depth_lr, mask_lr, gt_depth_lr, gt_mask_lr = downsample_data(
        rgb, depth, mask, gt_depth, gt_mask, 224, 224
    )

    # Save downsampled
    save_vis("ds224_rgb", rgb_lr, "downsample_vis")
    save_vis("ds224_depth", depth_lr, "downsample_vis")
    save_vis("ds224_mask", mask_lr, "downsample_vis", cmap='gray')
    save_vis("ds224_gt_depth", gt_depth_lr, "downsample_vis")
    save_vis("ds224_gt_mask", gt_mask_lr, "downsample_vis", cmap='gray')

    # Stats
    print(f"\nOriginal depth valid: {mask.sum():.0f} pixels")
    print(f"Downsampled depth valid: {mask_lr.sum():.0f} pixels")
    print(f"Original GT valid: {gt_mask.sum():.0f} pixels")
    print(f"Downsampled GT valid: {gt_mask_lr.sum():.0f} pixels")

    print("\nDone! Check downsample_vis/")

if __name__ == "__main__":
    main()