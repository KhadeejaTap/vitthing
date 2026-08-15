#!/usr/bin/env python3
"""Save center-cropped 224x224 frame_0000 data as .npy files."""
import numpy as np
from PIL import Image
import os

def center_crop(arr, target_h, target_w):
    """Crop center region of array."""
    h, w = arr.shape[:2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    if arr.ndim == 3:
        return arr[top:top+target_h, left:left+target_w, :]
    else:
        return arr[top:top+target_h, left:left+target_w]

def main():
    os.makedirs("data_crop224", exist_ok=True)

    print("Loading original 540x960...")
    rgb = np.array(Image.open('data/frame_0000_rgb.png')).astype(np.float32) / 255.0
    depth = np.load('data/frame_0000_depth_proj_mm.npy').astype(np.float32)
    mask = np.load('data/frame_0000_proj_valid_mask.npy').astype(np.float32)
    gt_depth = np.load('data/frame_0000_gt_mm.npy').astype(np.float32)
    gt_mask = np.load('data/frame_0000_gt_mask.npy').astype(np.float32)

    # Center crop to 224x224
    print("Center cropping to 224x224...")
    rgb_cc = center_crop(rgb, 224, 224)
    depth_cc = center_crop(depth, 224, 224)
    mask_cc = center_crop(mask, 224, 224)
    gt_depth_cc = center_crop(gt_depth, 224, 224)
    gt_mask_cc = center_crop(gt_mask, 224, 224)

    # Save as .npy
    np.save('data_crop224/frame_0000_rgb.npy', rgb_cc)
    np.save('data_crop224/frame_0000_depth_proj_mm.npy', depth_cc)
    np.save('data_crop224/frame_0000_proj_valid_mask.npy', mask_cc)
    np.save('data_crop224/frame_0000_gt_mm.npy', gt_depth_cc)
    np.save('data_crop224/frame_0000_gt_mask.npy', gt_mask_cc)

    print(f"Saved to data_crop224/:")
    print(f"  rgb: {rgb_cc.shape} (float32, 0-1)")
    print(f"  depth_proj_mm: {depth_cc.shape} (float32, mm)")
    print(f"  proj_valid_mask: {mask_cc.shape} (float32, 0/1)")
    print(f"  gt_mm: {gt_depth_cc.shape} (float32, mm)")
    print(f"  gt_mask: {gt_mask_cc.shape} (float32, 0/1)")

    # Stats
    print(f"\nCropped depth valid: {mask_cc.sum():.0f} pixels")
    print(f"Cropped GT valid: {gt_mask_cc.sum():.0f} pixels")
    print(f"Depth range: {depth_cc[mask_cc>0].min():.1f} - {depth_cc[mask_cc>0].max():.1f} mm")
    print(f"GT depth range: {gt_depth_cc[gt_mask_cc>0].min():.1f} - {gt_depth_cc[gt_mask_cc>0].max():.1f} mm")

if __name__ == "__main__":
    main()