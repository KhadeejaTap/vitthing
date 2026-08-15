#!/usr/bin/env python3
"""
Hypersim Dataset for dToF training.

Wraps the augmentation pipeline and outputs tensors compatible with the model:
- rgb: (3, H, W) in [0,1]
- depth_input: (3, H, W) in [-1,1] from build_input_tensor
- valid_mask: (H, W) binary (only real sensor measurements)
- gt_depth: (1, H, W) in mm
- gt_mask: (1, H, W) binary
"""

import random
from typing import Optional
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from augment import apply_augmentation_pipeline
    from normalize import compute_log_params, build_input_tensor
except ImportError:  # fallback when imported as code.hypersim_dataset
    from .augment import apply_augmentation_pipeline
    from .normalize import compute_log_params, build_input_tensor


# Hypersim camera intrinsics
HYPERSIM_HEIGHT = 768
HYPERSIM_WIDTH = 1024
HYPERSIM_FOCAL = 886.81

# Precompute image plane for distance-to-depth conversion
_imageplane_x = np.linspace((-0.5 * HYPERSIM_WIDTH) + 0.5, (0.5 * HYPERSIM_WIDTH) - 0.5, HYPERSIM_WIDTH).reshape(1, HYPERSIM_WIDTH).repeat(HYPERSIM_HEIGHT, 0).astype(np.float32)[:, :, None]
_imageplane_y = np.linspace((-0.5 * HYPERSIM_HEIGHT) + 0.5, (0.5 * HYPERSIM_HEIGHT) - 0.5, HYPERSIM_HEIGHT).reshape(HYPERSIM_HEIGHT, 1).repeat(HYPERSIM_WIDTH, 1).astype(np.float32)[:, :, None]
_imageplane_z = np.full([HYPERSIM_HEIGHT, HYPERSIM_WIDTH, 1], HYPERSIM_FOCAL, np.float32)
_IMAGEPLANE = np.concatenate([_imageplane_x, _imageplane_y, _imageplane_z], 2)
_IMAGEPLANE_NORM = np.linalg.norm(_IMAGEPLANE, 2, 2)


def distance_to_depth(distance: np.ndarray) -> np.ndarray:
    """Convert Euclidean distance to planar depth (z-depth)."""
    return distance / _IMAGEPLANE_NORM * HYPERSIM_FOCAL


def build_hypersim_index(scenes_root: str):
    """Walk scenes_root and build flat list of (color_path, depth_path) pairs."""
    samples = []
    scenes_root = Path(scenes_root)

    for scene_dir in sorted(scenes_root.glob("ai_*")):
        images_dir = scene_dir / "images"
        if not images_dir.is_dir():
            continue

        final_dirs = sorted(images_dir.glob("scene_cam_*_final_hdf5"))
        for final_dir in final_dirs:
            cam_name = final_dir.name.replace("_final_hdf5", "")
            geometry_dir = images_dir / f"{cam_name}_geometry_hdf5"
            if not geometry_dir.is_dir():
                continue

            for color_file in sorted(final_dir.glob("frame.*.color.hdf5")):
                frame_id = color_file.name.split(".")[1]
                depth_file = geometry_dir / f"frame.{frame_id}.depth_meters.hdf5"
                if depth_file.exists():
                    samples.append({
                        "scene": scene_dir.name,
                        "cam": cam_name,
                        "frame": frame_id,
                        "color_path": str(color_file),
                        "depth_path": str(depth_file),
                    })

    return samples


def split_by_scene(samples, val_fraction=0.05):
    """Split samples by scene: last val_fraction alphabetically go to validation."""
    scenes = sorted(set(s["scene"] for s in samples))
    n_val = max(1, int(len(scenes) * val_fraction))
    val_scenes = set(scenes[-n_val:])
    train = [s for s in samples if s["scene"] not in val_scenes]
    val = [s for s in samples if s["scene"] in val_scenes]
    return train, val, sorted(val_scenes)


def split_by_frame(samples, val_fraction=0.05):
    """Split samples by frame count: last val_fraction go to validation."""
    samples = sorted(samples, key=lambda s: (s["scene"], s["cam"], int(s["frame"])))
    n_val = max(1, int(len(samples) * val_fraction))
    val = samples[-n_val:]
    train = samples[:-n_val]
    return train, val, n_val


def split_train_stages(samples):
    """Split training samples into 0-49%, 50-74%, 75-100% frame bands."""
    n = len(samples)
    b1 = n // 2
    b2 = (3 * n) // 4
    return [samples[:b1], samples[b1:b2], samples[b2:]]


def load_sample(color_path: str, depth_path: str):
    """Load color and depth HDF5 files, return dict with arrays and metadata."""
    with h5py.File(color_path, "r") as f:
        color_keys = list(f.keys())
        color_data = {}
        for k in color_keys:
            arr = f[k][:]
            color_data[k] = arr

    with h5py.File(depth_path, "r") as f:
        depth_keys = list(f.keys())
        depth_data = {}
        for k in depth_keys:
            arr = f[k][:]
            depth_data[k] = arr

    # Convert depth from Euclidean distance to planar z-depth
    euclidean_distance = None
    if "dataset" in depth_data:
        euclidean_distance = depth_data["dataset"].copy()
        depth_data["dataset"] = distance_to_depth(depth_data["dataset"])

    return {
        "color_keys": color_keys,
        "color_data": color_data,
        "depth_keys": depth_keys,
        "depth_data": depth_data,
        "euclidean_distance": euclidean_distance,
    }


class HypersimDataset(Dataset):
    """
    PyTorch Dataset for Hypersim with sensor-to-RGB reprojection augmentation.

    Outputs per sample:
        - rgb: (3, H, W) float32 in [0,1]
        - depth_input: (3, H, W) float32 in [-1,1] (log-encoded + mask)
        - valid_mask: (H, W) float32 binary (only real sensor measurements)
        - gt_depth: (1, H, W) float32 in mm
        - gt_mask: (1, H, W) float32 binary
    """

    def __init__(
        self,
        scenes_root: str,
        stage: int = 1,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 42,
        sparsity_min: float = 0.2,
        sparsity_max: float = 0.3,
        noise_std: float = 0.01,
        extrinsics_trans_px: float = 0.5,
        extrinsics_rot_deg: float = 0.1,
        samples: Optional[list] = None,
    ):
        """
        Args:
            scenes_root: Path to scenes root (e.g. .../evermotion_dataset/scenes)
            stage: Training stage (1, 2, 3, 4) - determines resolution
            split: "train" or "val"
            val_fraction: Fraction of scenes for validation (last N alphabetically)
            seed: Random seed for reproducibility
            sparsity_min: Minimum sparsity ratio
            sparsity_max: Maximum sparsity ratio
            noise_std: Gaussian noise std for depth (meters)
            extrinsics_trans_px: Extrinsic translation error std (pixels)
            extrinsics_rot_deg: Extrinsic rotation error std (degrees)
            samples: Pre-built sample list (for val set sharing train's index)
        """
        self.stage = stage
        self.split = split
        self.sparsity_min = sparsity_min
        self.sparsity_max = sparsity_max
        self.noise_std = noise_std
        self.extrinsics_trans_px = extrinsics_trans_px
        self.extrinsics_rot_deg = extrinsics_rot_deg
        self.seed = seed

        # Log normalization params (fixed sensor range)
        self.alpha, self.beta = compute_log_params()

        # Build or use provided index
        if samples is not None:
            self.samples = samples
        else:
            all_samples = build_hypersim_index(scenes_root)
            train_samples, val_samples, _ = split_by_scene(all_samples, val_fraction)
            self.samples = train_samples if split == "train" else val_samples

        # Random generator per worker
        self.rng = np.random.default_rng(seed)
        self._rng_worker_id = None

        print(f"HypersimDataset: split={split}, stage={stage}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.id != self._rng_worker_id:
            self.rng = np.random.default_rng(self.seed + worker_info.id)
            self._rng_worker_id = worker_info.id

        sample = self.samples[idx]

        # Load raw data
        loaded = load_sample(sample["color_path"], sample["depth_path"])

        color_key = loaded["color_keys"][0]
        depth_key = loaded["depth_keys"][0]

        rgb = loaded["color_data"][color_key].astype(np.float32)  # (H, W, 3) in [0,1] HDR
        gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # (H, W) planar z-depth in meters
        euclidean_distance = loaded.get("euclidean_distance", None)
        if euclidean_distance is not None:
            euclidean_distance = euclidean_distance.astype(np.float32)

        # Apply augmentation pipeline
        sparsity = self.rng.uniform(self.sparsity_min, self.sparsity_max)
        out = apply_augmentation_pipeline(
            rgb, gt_depth, self.stage, sparsity, self.noise_std, self.rng,
            extrinsics_trans_px=self.extrinsics_trans_px,
            extrinsics_rot_deg=self.extrinsics_rot_deg,
            euclidean_distance=euclidean_distance
        )

        # Extract outputs
        rgb_out = out["rgb"]                    # (H, W, 3) in [0,1]
        gt_depth_out = out["gt_depth"]          # (H, W) planar z-depth in meters
        gt_valid_mask = out["gt_valid_mask"]    # (H, W) binary
        reprojected_depth = out["reprojected_depth"]  # (H, W) planar z-depth in meters
        reprojected_mask = out["reprojected_mask"]    # (H, W) binary (only real measurements)

        # Convert gt_depth to mm
        gt_depth_mm = gt_depth_out * 1000.0

        # Build depth_input tensor (3, H, W) in [-1,1]
        depth_input = build_input_tensor(reprojected_depth * 1000.0, reprojected_mask, self.alpha, self.beta)

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_out).permute(2, 0, 1).contiguous()  # (3, H, W)
        depth_input_tensor = torch.from_numpy(depth_input).contiguous()        # (3, H, W)
        valid_mask_tensor = torch.from_numpy(reprojected_mask).contiguous()    # (H, W)
        gt_depth_tensor = torch.from_numpy(gt_depth_mm).unsqueeze(0).contiguous()  # (1, H, W)
        gt_mask_tensor = torch.from_numpy(gt_valid_mask).unsqueeze(0).contiguous()  # (1, H, W)

        return {
            "rgb": rgb_tensor,
            "depth_input": depth_input_tensor,
            "valid_mask": valid_mask_tensor,
            "gt_depth": gt_depth_tensor,
            "gt_mask": gt_mask_tensor,
            "alpha": self.alpha,
            "beta": self.beta,
            "stage": self.stage,
            "sparsity": sparsity,
            "scene": sample["scene"],
            "cam": sample["cam"],
            "frame": sample["frame"],
        }


def get_dataloaders(
    scenes_root: str,
    stage: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    val_fraction: float = 0.05,
    seed: int = 42,
    **dataset_kwargs
):
    """Create train and val dataloaders."""
    # Build shared index
    all_samples = build_hypersim_index(scenes_root)
    train_samples, val_samples, _ = split_by_frame(all_samples, val_fraction)

    train_ds = HypersimDataset(
        scenes_root, stage=stage, split="train", val_fraction=val_fraction,
        seed=seed, samples=train_samples, **dataset_kwargs
    )
    val_ds = HypersimDataset(
        scenes_root, stage=stage, split="val", val_fraction=val_fraction,
        seed=seed, samples=val_samples, **dataset_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test - use relative path or env var
    import os
    scenes_root = os.environ.get("HYPERSIM_ROOT", "/home/khadeeja/ml-hypersim/evermotion_dataset/scenes")
    ds = HypersimDataset(
        scenes_root,
        stage=1,
        split="train",
    )
    print(f"Dataset length: {len(ds)}")

    sample = ds[0]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.shape} {v.dtype} [{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {k}: {v}")