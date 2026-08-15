#!/usr/bin/env python3
"""Test and visualize sensor mask augmentations."""
import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from code.dataset import (
    load_hypersim_frame,
    create_gt_depth_mask,
    create_sensor_mask,
    combine_masks,
    build_input_tensor,
    HypersimDepthCompletionDataset,
    TARGET_RESOLUTIONS,
    RESOLUTION_WEIGHTS,
)

scene_dir = Path("/home/khadeeja/ml-hypersim/evermotion_dataset/scenes/ai_001_001")
frame = load_hypersim_frame(scene_dir, frame_idx=0)
depth_mm = frame['depth_mm']
valid_mask = frame['valid_mask']
h, w = depth_mm.shape

gt_mask = create_gt_depth_mask(depth_mm, valid_mask)

print("=" * 60)
print("Testing Sensor Mask Augmentations (Realistic Reprojection)")
print("=" * 60)

# Test 1: FOV mismatch + sensor position jitter
print("\n1. FOV Mismatch + Sensor Position Jitter (4:3 sensor in 16:9 frame)")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    mask = create_sensor_mask(h, w,
        sensor_h=480, sensor_w=640, sensor_fx=544.462653,
        rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
        sparsity_level="low",
        sensor_pos_jitter=0.15,
        edge_falloff=0.0)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Sample {i+1} (valid: {mask.mean()*100:.1f}%)')
    ax.axis('off')
plt.suptitle('Sensor FOV (60°) in RGB FOV (90°) + Random Position')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_fov_jitter.png', dpi=150)
print("  Saved: test_out/aug_fov_jitter.png")

# Test 2: Sparsity levels
print("\n2. Sparsity Levels (low/medium/high)")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, level in enumerate(["low", "medium", "high"]):
    mask = create_sensor_mask(h, w,
        sensor_h=480, sensor_w=640, sensor_fx=544.462653,
        rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
        sparsity_level=level,
        sensor_pos_jitter=0.15,
        edge_falloff=0.1)
    axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f'{level} (valid: {mask.mean()*100:.1f}%)')
    axes[i].axis('off')
plt.suptitle('Sparsity Levels')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_sparsity_levels.png', dpi=150)
print("  Saved: test_out/aug_sparsity_levels.png")

# Test 3: Edge falloff
print("\n3. Edge Falloff (within sensor region)")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, falloff in enumerate([0.0, 0.2, 0.4]):
    mask = create_sensor_mask(h, w,
        sensor_h=480, sensor_w=640, sensor_fx=544.462653,
        rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
        sparsity_level="medium",
        sensor_pos_jitter=0.15,
        edge_falloff=falloff)
    axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f'falloff={falloff} (valid: {mask.mean()*100:.1f}%)')
    axes[i].axis('off')
plt.suptitle('Edge Falloff')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_edge_falloff.png', dpi=150)
print("  Saved: test_out/aug_edge_falloff.png")

# Test 4: Depth-dependent sparsity
print("\n4. Depth-Dependent Sparsity (increases with depth)")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, scale in enumerate([0.0, 0.3, 0.6]):
    mask = create_sensor_mask(h, w,
        sensor_h=480, sensor_w=640, sensor_fx=544.462653,
        rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
        sparsity_level="medium",
        sensor_pos_jitter=0.15,
        edge_falloff=0.1,
        depth_mm=depth_mm,
        depth_sparsity_scale=scale)
    axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f'depth_scale={scale} (valid: {mask.mean()*100:.1f}%)')
    axes[i].axis('off')
plt.suptitle('Depth-Dependent Sparsity')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_depth_sparsity.png', dpi=150)
print("  Saved: test_out/aug_depth_sparsity.png")

# Test 5: Combined all effects
print("\n5. Combined: All Effects")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    mask = create_sensor_mask(h, w,
        sensor_h=480, sensor_w=640, sensor_fx=544.462653,
        rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
        sparsity_level="random",
        sensor_pos_jitter=0.15,
        edge_falloff=0.2,
        depth_mm=depth_mm,
        depth_sparsity_scale=0.3)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Sample {i+1} (valid: {mask.mean()*100:.1f}%)')
    ax.axis('off')
plt.suptitle('All Effects Combined')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_combined.png', dpi=150)
print("  Saved: test_out/aug_combined.png")

# Test 6: Multi-resolution sampling
print("\n6. Multi-Resolution Sampling")
dataset = HypersimDepthCompletionDataset(
    scene_dir, frame_indices=[0],
    target_resolution=(224, 224),
    multi_res=True,
    sensor_h=480, sensor_w=640, sensor_fx=544.462653,
    rgb_h=1080, rgb_w=1920, rgb_fx=910.799450,
    sparsity_level="random",
    sensor_pos_jitter=0.15,
    edge_falloff=0.2,
    depth_sparsity_scale=0.3,
)
res_counts = {}
for i in range(50):
    sample = dataset[0]
    res = f"{sample['rgb'].shape[1]}x{sample['rgb'].shape[2]}"
    res_counts[res] = res_counts.get(res, 0) + 1
print("  Resolution distribution (50 samples):")
for res, count in sorted(res_counts.items()):
    print(f"    {res}: {count}")

# Test 7: Full pipeline visualization
print("\n7. Full Pipeline: RGB, Depth Input, GT, Sensor Mask")
sample = dataset[0]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(sample['rgb'].permute(1, 2, 0).numpy())
axes[0, 0].set_title('RGB')
axes[0, 0].axis('off')

axes[0, 1].imshow(sample['depth_input'][0].numpy(), cmap='turbo')
axes[0, 1].set_title('Log Depth Norm (ch0)')
axes[0, 1].axis('off')

axes[0, 2].imshow(sample['depth_input'][2].numpy(), cmap='gray')
axes[0, 2].set_title('Sensor Mask (ch2)')
axes[0, 2].axis('off')

axes[1, 0].imshow(sample['gt_depth'][0].numpy(), cmap='turbo')
axes[1, 0].set_title('GT Depth (mm)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sample['gt_mask'][0].numpy(), cmap='gray')
axes[1, 1].set_title('GT Mask (300-8333mm)')
axes[1, 1].axis('off')

axes[1, 2].imshow(sample['sensor_mask'][0].numpy(), cmap='gray')
axes[1, 2].set_title('Sensor Mask (raw)')
axes[1, 2].axis('off')

plt.suptitle(f'Full Sample: {sample["rgb"].shape[1]}x{sample["rgb"].shape[2]}')
plt.tight_layout()
plt.savefig('/home/khadeeja/vitthing/test_out/aug_full_pipeline.png', dpi=150)
print("  Saved: test_out/aug_full_pipeline.png")

print("\n" + "=" * 60)
print("All augmentation tests complete!")
print("=" * 60)