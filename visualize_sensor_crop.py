#!/usr/bin/env python3
"""
Visualize sensor crop overlay on GT depth for each stage.
Per-frame normalization to see subtle differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_root = Path("hypersim_data")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for stage_idx, stage in enumerate([1, 2, 3]):
    stage_dir = data_root / f"stage{stage}" / "train"
    files = sorted(stage_dir.glob("*.npz"))

    # Show first sample per stage
    data = np.load(files[0], allow_pickle=True)
    gt_depth = data["gt_depth"]  # (H, W) mm
    sensor_depth = data["sensor_depth"]  # (sh, sw) mm
    crop_perimeter = data["crop_perimeter"]  # [y, x, h, w]
    meta = data["meta"]

    crop_y, crop_x, crop_h, crop_w = crop_perimeter

    # Per-frame normalization: use each frame's own min/max (excluding zeros/invalid)
    gt_valid = gt_depth[gt_depth > 0]
    sensor_valid = sensor_depth[sensor_depth > 0]
    gt_vmin, gt_vmax = gt_valid.min(), gt_valid.max()
    sensor_vmin, sensor_vmax = sensor_valid.min(), sensor_valid.max()

    # 1. GT Depth with crop rectangle overlay (per-frame normalized)
    ax = axes[stage_idx, 0]
    im = ax.imshow(gt_depth, cmap="plasma", vmin=gt_vmin, vmax=gt_vmax)
    # Draw crop rectangle
    rect = plt.Rectangle((crop_x, crop_y), crop_w, crop_h,
                         linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Stage {stage} - GT Depth + Sensor Crop\n{meta[0]} {meta[1]} frame{meta[2]}\nRange: {gt_vmin:.0f}-{gt_vmax:.0f}mm")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")

    # 2. Sensor depth (cropped region, per-frame normalized)
    ax = axes[stage_idx, 1]
    im = ax.imshow(sensor_depth, cmap="plasma", vmin=sensor_vmin, vmax=sensor_vmax)
    ax.set_title(f"Sensor Depth (cropped)\n{crop_h}×{crop_w} at ({crop_y},{crop_x})\nRange: {sensor_vmin:.0f}-{sensor_vmax:.0f}mm")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")

    # 3. Sensor mask
    ax = axes[stage_idx, 2]
    sensor_mask = data["sensor_mask"]
    ax.imshow(sensor_mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Sensor Mask (valid pixels)\n{sensor_mask.mean()*100:.1f}% valid")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sensor_crop_visualization.png", dpi=150)
print("Saved to sensor_crop_visualization.png")
plt.show()