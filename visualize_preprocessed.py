#!/usr/bin/env python3
"""
Visualize preprocessed hypersim_data to verify correctness.
Shows RGB and GT depth for samples from each stage.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_root = Path("hypersim_data")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for stage_idx, stage in enumerate([1, 2, 3]):
    stage_dir = data_root / f"stage{stage}" / "train"
    files = sorted(stage_dir.glob("*.npz"))

    # Show first 2 samples per stage
    for sample_idx in range(min(2, len(files))):
        data = np.load(files[sample_idx], allow_pickle=True)
        rgb = data["rgb"]  # (H, W, 3) in [0,1]
        gt_depth = data["gt_depth"]  # (H, W) meters
        meta = data["meta"]

        col = sample_idx * 2

        # RGB
        ax = axes[stage_idx, col]
        ax.imshow(rgb)
        ax.set_title(f"Stage {stage} - Sample {sample_idx}\n{meta[0]} {meta[1]} frame{meta[2]}")
        ax.axis("off")

        # GT Depth
        ax = axes[stage_idx, col + 1]
        im = ax.imshow(gt_depth, cmap="plasma", vmin=0, vmax=8)
        ax.set_title(f"GT Depth (meters)\n{min(gt_depth.min(), 0):.2f} - {gt_depth.max():.2f}m")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("preprocessed_visualization.png", dpi=150)
print("Saved to preprocessed_visualization.png")
plt.show()