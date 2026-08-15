# Hypersim Data Preprocessing Plan

## Goal
Preprocess Hypersim data offline to avoid GPU walltime wasted on:
- Flood-fill (scipy distance_transform_edt) every batch
- Depth normalization/log-encoding
- Sensor simulation (sparsification, noise, reprojection)
- Padding to patch multiples

## Current Pipeline (per batch, on GPU)
1. Load raw HDF5 → RGB + depth
2. Downsample to stage resolution
3. Random crop to sensor resolution
4. Sparsify + add noise
5. Reproject to RGB grid (with extrinsic errors)
6. Flood-fill nearest neighbor (CPU scipy!)
7. Log-normalize depth → build 3-channel input tensor
8. Pad to 14× multiple
9. Forward pass

## Proposed Preprocessing (once, offline)

### Output Structure
```
data/
├── stage1/
│   ├── train/
│   │   ├── scene_cam_frame_0000.npz
│   │   └── ...
│   └── val/
│       └── ...
├── stage2/
│   ├── train/
│   └── val/
├── stage3/
│   ├── train/
│   └── val/
└── stage4/
    ├── train/
    └── val/
```

### Per-Sample NPZ Contents
```python
{
    "rgb": uint8[H, W, 3],           # Padded RGB in [0, 255]
    "depth_input": float32[3, H, W], # Pre-normalized [-1, 1] tensor (ch0=ch1=zhat, ch2=mask)
    "gt_depth": float32[1, H, W],    # Ground truth depth in mm
    "gt_mask": bool[1, H, W],        # Valid GT mask
    "valid_mask": bool[H, W],        # Sensor valid mask (for loss)
    "intrinsics": float32[4],        # fx, fy, cx, cy for this stage
    "meta": {
        "scene": str,
        "cam": str,
        "frame": int,
        "stage": int,
        "sparsity": float,
    }
}
```

### Preprocessing Script: `preprocess_hypersim.py`

```python
#!/usr/bin/env python3
"""
Offline preprocessing for Hypersim dataset.
Run once per stage to generate ready-to-train NPZ files.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import h5py
import cv2
from scipy.ndimage import distance_transform_edt

# Reuse existing modules
from code.hypersim_dataset import (
    build_hypersim_index,
    split_by_frame,
    split_train_stages,
    load_sample,
    apply_augmentation_pipeline,
)
from code.normalize import compute_log_params, build_input_tensor
from code.encoder import pad_to_multiple

STAGES = {
    1: {"divisor": 8, "sensor_h": 60, "sensor_w": 80},
    2: {"divisor": 5, "sensor_h": 96, "sensor_w": 128},
    3: {"divisor": 4, "sensor_h": 120, "sensor_w": 160},
    4: {"divisor": 2, "sensor_h": 240, "sensor_w": 320},
}

def preprocess_stage(scenes_root, stage, split, samples, out_dir, args, rng):
    """Preprocess all samples for a given stage/split."""
    alpha, beta = compute_log_params()
    stage_cfg = STAGES[stage]
    divisor = stage_cfg["divisor"]
    sensor_h, sensor_w = stage_cfg["sensor_h"], stage_cfg["sensor_w"]
    scale = 1.0 / divisor
    
    out_split_dir = out_dir / f"stage{stage}" / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, sample in enumerate(tqdm(samples, desc=f"Stage {stage} {split}")):
        # Load raw data
        loaded = load_sample(sample["color_path"], sample["depth_path"])
        color_key = loaded["color_keys"][0]
        depth_key = loaded["depth_keys"][0]
        
        rgb = loaded["color_data"][color_key].astype(np.float32)  # (H, W, 3) in [0,1]
        gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # (H, W) meters
        euclidean_distance = loaded.get("euclidean_distance", None)
        if euclidean_distance is not None:
            euclidean_distance = euclidean_distance.astype(np.float32)
        
        # For training: use multiple sparsity values per frame (data augmentation)
        # For validation: use fixed sparsity
        if split == "train":
            sparsity_values = np.linspace(args.sparsity_min, args.sparsity_max, 3)
        else:
            sparsity_values = [(args.sparsity_min + args.sparsity_max) / 2]
        
        for sparsity in sparsity_values:
            # Apply augmentation pipeline (deterministic per sample+sparsity)
            sample_rng = np.random.default_rng(args.seed + hash((sample["scene"], sample["cam"], sample["frame"], sparsity)) % 2**32)
            
            out = apply_augmentation_pipeline(
                rgb, gt_depth, stage, sparsity, args.noise_std, sample_rng,
                extrinsics_trans_px=args.extrinsics_trans_px,
                extrinsics_rot_deg=args.extrinsics_rot_deg,
                euclidean_distance=euclidean_distance
            )
            
            # Extract outputs
            rgb_out = out["rgb"]                    # (H, W, 3) in [0,1]
            gt_depth_out = out["gt_depth"]          # (H, W) meters
            gt_valid_mask = out["gt_valid_mask"]    # (H, W) binary
            reprojected_depth = out["reprojected_depth"]  # (H, W) meters
            reprojected_mask = out["reprojected_mask"]    # (H, W) binary
            
            # Convert gt_depth to mm
            gt_depth_mm = gt_depth_out * 1000.0
            
            # Build depth_input tensor (3, H, W) in [-1,1] - includes flood-fill!
            depth_input = build_input_tensor(reprojected_depth * 1000.0, reprojected_mask, alpha, beta)
            
            # Pad to patch multiple (14)
            rgb_padded = pad_to_multiple(torch.from_numpy(rgb_out).permute(2, 0, 1).unsqueeze(0))
            depth_input_padded = pad_to_multiple(torch.from_numpy(depth_input).unsqueeze(0))
            gt_depth_padded = pad_to_multiple(torch.from_numpy(gt_depth_mm).unsqueeze(0).unsqueeze(0))
            gt_mask_padded = pad_to_multiple(torch.from_numpy(gt_valid_mask).unsqueeze(0).unsqueeze(0).float())
            valid_mask_padded = pad_to_multiple(torch.from_numpy(reprojected_mask).unsqueeze(0).unsqueeze(0).float())
            
            # Remove batch dim
            rgb_padded = rgb_padded.squeeze(0).numpy()      # (3, H_pad, W_pad)
            depth_input_padded = depth_input_padded.squeeze(0).numpy()  # (3, H_pad, W_pad)
            gt_depth_padded = gt_depth_padded.squeeze(0).numpy()        # (1, H_pad, W_pad)
            gt_mask_padded = gt_mask_padded.squeeze(0).numpy()          # (1, H_pad, W_pad)
            valid_mask_padded = valid_mask_padded.squeeze(0).numpy()    # (1, H_pad, W_pad)
            
            # Compute intrinsics for padded resolution
            H_pad, W_pad = rgb_padded.shape[-2:]
            focal = 886.81 * scale
            fx = fy = focal
            cx = (W_pad - 1) / 2.0
            cy = (H_pad - 1) / 2.0
            
            # Save as NPZ (compressed)
            sparsity_str = f"{sparsity:.3f}".replace(".", "p")
            fname = f"{sample['scene']}_{sample['cam']}_frame{sample['frame']}_sparsity{sparsity_str}.npz"
            np.savez_compressed(
                out_split_dir / fname,
                rgb=rgb_padded.astype(np.uint8),           # Save as uint8 [0,255]
                depth_input=depth_input_padded.astype(np.float32),
                gt_depth=gt_depth_padded.astype(np.float32),
                gt_mask=gt_mask_padded.astype(np.bool_),
                valid_mask=valid_mask_padded.squeeze(0).astype(np.bool_),
                intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
                meta=np.array([sample["scene"], sample["cam"], sample["frame"], stage, sparsity], dtype=object),
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity-min", type=float, default=0.2)
    parser.add_argument("--sparsity-max", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--extrinsics-trans-px", type=float, default=0.5)
    parser.add_argument("--extrinsics-rot-deg", type=float, default=0.1)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    args = parser.parse_args()
    
    print("Building dataset index...")
    all_samples = build_hypersim_index(args.scenes_root)
    train_samples, val_samples, _ = split_by_frame(all_samples, args.val_fraction)
    stage_samples = split_train_stages(train_samples)
    
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    
    for stage in args.stages:
        for split in args.splits:
            if split == "train":
                samples = stage_samples[stage - 1]
            else:
                samples = val_samples
            preprocess_stage(args.scenes_root, stage, split, samples, out_dir, args, rng)
    
    print(f"Done! Preprocessed data saved to {out_dir}")

if __name__ == "__main__":
    main()
```

## Modified Dataset for Training

```python
class PreprocessedHypersimDataset(Dataset):
    """Loads preprocessed NPZ files - no augmentation, no flood-fill, no padding."""
    
    def __init__(self, data_dir, stage, split):
        self.files = sorted(Path(data_dir).glob(f"stage{stage}/{split}/*.npz"))
        self.stage = stage
        self.split = split
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            "rgb": torch.from_numpy(data["rgb"]).float() / 255.0,  # (3, H, W) in [0,1]
            "depth_input": torch.from_numpy(data["depth_input"]),   # (3, H, W) in [-1,1]
            "gt_depth": torch.from_numpy(data["gt_depth"]),         # (1, H, W) mm
            "gt_mask": torch.from_numpy(data["gt_mask"]),           # (1, H, W) bool
            "valid_mask": torch.from_numpy(data["valid_mask"]),     # (H, W) bool
            "fx": float(data["intrinsics"][0]),
            "fy": float(data["intrinsics"][1]),
            "cx": float(data["intrinsics"][2]),
            "cy": float(data["intrinsics"][3]),
            "meta": data["meta"].tolist(),
        }
```

## Training Loop Changes

1. **Remove** flood-fill from `run_epoch` (lines 238-273)
2. **Remove** `pad_to_multiple` calls (lines 233-234)
3. **Use** intrinsics from batch instead of `compute_intrinsics()`
4. **DataLoader** can use `shuffle=True` since no per-epoch augmentation

## Expected Speedup

| Operation | Current (per batch) | Preprocessed |
|-----------|---------------------|--------------|
| HDF5 I/O | Yes | No (NPZ) |
| Downsample | Yes | No |
| Sparsify | Yes | No |
| Reproject | Yes | No |
| Flood-fill (scipy) | **Yes (slow!)** | **No** |
| Normalize | Yes | No |
| Pad | Yes | No |
| **Total overhead** | **~200-500ms/batch** | **~5-10ms/batch** |

## Usage

```bash
# One-time preprocessing (run on CPU machine, can parallelize)
python preprocess_hypersim.py \
    --scenes-root /home/khadeeja/ml-hypersim/evermotion_dataset/scenes \
    --out-dir data \
    --stages 1 2 3 4 \
    --splits train val

# Training (now much faster)
python code/train_staged.py \
    --data-dir data \
    --stage 1 \
    ...
```