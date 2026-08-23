#!/usr/bin/env python3
"""
Test script to build the dataset and print all information like labels, channels, etc.
"""

import torch
import numpy as np
from pathlib import Path
from main.preprocessed_dataset import PreprocessedHypersimDataset

def main():
    print("=== Dataset Information Test ===")

    # Set up dataset
    data_dir = str(Path(__file__).resolve().parent.parent / "hypersim_data")
    print(f"Data directory: {data_dir}")

    try:
        # Create dataset
        ds = PreprocessedHypersimDataset(data_dir=data_dir, stage=1, split="train")
        print(f"\nDataset length: {len(ds)}")

        if len(ds) == 0:
            print("ERROR: No samples found in dataset")
            return

        # Get first sample
        print("\n--- Loading first sample ---")
        sample = ds[0]

        # Print all sample information
        print("\n=== Sample Information ===")
        for k, v in sample.items():
            if torch.is_tensor(v):
                shape_str = str(list(v.shape))
                print(f"{k:20}: shape={shape_str:15} dtype={str(v.dtype):10} "
                      f"min={v.min().item():8.4f} max={v.max().item():8.4f}")
            elif isinstance(v, (list, tuple)):
                print(f"{k:20}: {v}")
            else:
                print(f"{k:20}: {v}")

        # Print specific information requested
        print("\n=== Requested Information ===")
        print(f"Labels (scene/cam/frame): {sample['scene']}/{sample['cam']}/{sample['frame']}")
        print(f"Stage: {sample['stage']}")
        print(f"Sensor option: {sample['sensor_option']}")
        print(f"Crop perimeter: {sample['crop_perimeter']}")

        # Channel information
        print(f"\n=== Channel Information ===")
        print(f"RGB channels: {sample['rgb'].shape[0]} (shape: {sample['rgb'].shape})")
        print(f"Depth filled mm channels: {sample['depth_filled_mm'].shape[0]} (shape: {sample['depth_filled_mm'].shape})")
        print(f"Valid mask channels: {sample['valid_mask'].shape[0]} (shape: {sample['valid_mask'].shape})")
        print(f"GT depth channels: {sample['gt_depth'].shape[0]} (shape: {sample['gt_depth'].shape})")
        print(f"GT mask channels: {sample['gt_mask'].shape[0]} (shape: {sample['gt_mask'].shape})")

        # Intrinsics
        print(f"\n=== Camera Intrinsics ===")
        print(f"fx: {sample['fx']:.2f}")
        print(f"fy: {sample['fy']:.2f}")
        print(f"cx: {sample['cx']:.2f}")
        print(f"cy: {sample['cy']:.2f}")

        # Value ranges
        print(f"\n=== Value Ranges ===")
        print(f"RGB range: [{sample['rgb'].min().item():.4f}, {sample['rgb'].max().item():.4f}]")
        print(f"Depth filled mm range: [{sample['depth_filled_mm'].min().item():.2f}, {sample['depth_filled_mm'].max().item():.2f}] mm")
        print(f"GT depth range: [{sample['gt_depth'].min().item():.2f}, {sample['gt_depth'].max().item():.2f}] mm")
        print(f"Valid mask unique values: {torch.unique(sample['valid_mask'].int()).tolist()}")
        print(f"GT mask unique values: {torch.unique(sample['gt_mask'].int()).tolist()}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()