#!/usr/bin/env python3
"""
Augmentation utilities for sensor-to-RGB reprojection simulation.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional

# Depth range limits (meters)
DEPTH_MIN = 0.3
DEPTH_MAX = 8.333


@dataclass
class CropPerimeter:
    """Stores crop information for reprojection."""
    x: int      # top-left x in high-res coordinates
    y: int      # top-left y in high-res coordinates
    w: int      # width of crop in high-res coordinates
    h: int      # height of crop in high-res coordinates
    scale: float  # scale factor from sensor to high-res


def apply_depth_range_mask(depth: np.ndarray) -> np.ndarray:
    """Create valid mask for depth within [DEPTH_MIN, DEPTH_MAX]."""
    return (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)


def random_crop(image: np.ndarray, crop_h: int, crop_w: int, rng: np.random.Generator) -> Tuple[np.ndarray, CropPerimeter]:
    """
    Randomly crop image to (crop_h, crop_w). Returns cropped image and perimeter info.
    """
    H, W = image.shape[:2]
    assert crop_h <= H and crop_w <= W, f"Crop {crop_h}x{crop_w} larger than image {H}x{W}"

    y = rng.integers(0, H - crop_h + 1)
    x = rng.integers(0, W - crop_w + 1)

    cropped = image[y:y+crop_h, x:x+crop_w].copy()
    perimeter = CropPerimeter(x=x, y=y, w=crop_w, h=crop_h, scale=1.0)
    return cropped, perimeter


def sparsify_depth(depth: np.ndarray, sparsity: float, rng: np.random.Generator,
                   noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparsify depth map: keep only `sparsity` fraction of pixels within valid depth range, add Gaussian noise.
    Returns (sparse_depth, valid_mask).
    """
    H, W = depth.shape

    # Valid pixels: within depth range AND random sparsity
    range_mask = apply_depth_range_mask(depth)
    random_mask = rng.random((H, W)) < sparsity
    valid_mask = range_mask & random_mask

    sparse_depth = np.zeros_like(depth)
    sparse_depth[valid_mask] = depth[valid_mask]

    # Add Gaussian noise to valid pixels only
    if noise_std > 0:
        noise = rng.normal(0, noise_std, size=valid_mask.sum())
        sparse_depth[valid_mask] += noise
        # Ensure depth stays positive
        sparse_depth[valid_mask] = np.maximum(sparse_depth[valid_mask], 0.001)

    return sparse_depth, valid_mask.astype(np.float32)


def reproject_nearest(sparse_depth: np.ndarray, valid_mask: np.ndarray,
                      perimeter: CropPerimeter, target_h: int, target_w: int,
                      rng: np.random.Generator = None,
                      extrinsics_trans_px: float = 0.0,
                      extrinsics_rot_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproject sparse low-res depth to high-res target using nearest neighbor.

    Two-step process:
    1. Reproject: scatter sparse points to their correct positions in full RGB grid
    2. Flood fill nearest neighbor: fill entire target from scattered points using scipy's distance_transform_edt

    Args:
        extrinsics_trans_px: Translation error in pixels (std dev for x,y)
        extrinsics_rot_deg: Rotation error in degrees (std dev)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create high-res canvas
    reprojected = np.zeros((target_h, target_w), dtype=np.float32)
    reprojected_mask = np.zeros((target_h, target_w), dtype=np.float32)

    # Sample extrinsic calibration error
    tx = rng.normal(0, extrinsics_trans_px) if extrinsics_trans_px > 0 else 0.0
    ty = rng.normal(0, extrinsics_trans_px) if extrinsics_trans_px > 0 else 0.0
    rot_rad = np.deg2rad(rng.normal(0, extrinsics_rot_deg)) if extrinsics_rot_deg > 0 else 0.0

    # Scale perimeter to target resolution
    x1 = int(perimeter.x * perimeter.scale)
    y1 = int(perimeter.y * perimeter.scale)
    x2 = int((perimeter.x + perimeter.w) * perimeter.scale)
    y2 = int((perimeter.y + perimeter.h) * perimeter.scale)

    # Apply translation error
    x1 += int(round(tx))
    y1 += int(round(ty))
    x2 += int(round(tx))
    y2 += int(round(ty))

    # Apply rotation error around crop center
    if rot_rad != 0:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)

        def rotate(x, y):
            x_rel, y_rel = x - cx, y - cy
            x_new = cx + cos_r * x_rel - sin_r * y_rel
            y_new = cy + sin_r * x_rel + cos_r * y_rel
            return x_new, y_new

        x1, y1 = rotate(x1, y1)
        x2, y2 = rotate(x2, y2)
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))

    # Clamp to target bounds
    x1 = max(0, min(x1, target_w - 1))
    y1 = max(0, min(y1, target_h - 1))
    x2 = max(x1 + 1, min(x2, target_w))
    y2 = max(y1 + 1, min(y2, target_h))

    crop_h = y2 - y1
    crop_w = x2 - x1

    if crop_h <= 0 or crop_w <= 0:
        return reprojected, reprojected_mask

    # Step 1: REPROJECT - scatter sparse points to their positions in full RGB grid
    sparse_h, sparse_w = sparse_depth.shape
    valid_coords = np.argwhere(valid_mask > 0.5)

    if len(valid_coords) > 0:
        # Normalized coordinates in sparse crop [0, 1]
        sy = valid_coords[:, 0] / sparse_h
        sx = valid_coords[:, 1] / sparse_w

        # Map to target crop region
        ty = y1 + sy * crop_h
        tx = x1 + sx * crop_w

        # Round to nearest pixel in target grid
        ty = np.clip(np.round(ty).astype(int), y1, y2 - 1)
        tx = np.clip(np.round(tx).astype(int), x1, x2 - 1)

        # Scatter depth values
        reprojected[ty, tx] = sparse_depth[valid_coords[:, 0], valid_coords[:, 1]]
        reprojected_mask[ty, tx] = 1.0

    # Step 2: FLOOD FILL NEAREST NEIGHBOR using scipy's distance_transform_edt
    # Keep original scattered mask as validity (only real sensor measurements are valid)
    # Fill depth values for all pixels using nearest neighbor
    if np.any(reprojected_mask > 0.5):
        from scipy.ndimage import distance_transform_edt

        invalid = (reprojected_mask == 0)
        if invalid.any():
            # Get indices of nearest valid pixel for each invalid pixel
            idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
            reprojected = reprojected[tuple(idx)]
            # reprojected_mask stays as original scattered mask (only real measurements valid)

    return reprojected, reprojected_mask


def downsample_image(image: np.ndarray, scale: float, interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Downsample image by scale factor."""
    H, W = image.shape[:2]
    new_h, new_w = int(H * scale), int(W * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def apply_augmentation_pipeline(rgb: np.ndarray, gt_depth: np.ndarray,
                                 stage: int, sparsity: float, noise_std: float,
                                 rng: np.random.Generator,
                                 extrinsics_trans_px: float = 0.5,
                                 extrinsics_rot_deg: float = 0.1,
                                 euclidean_distance: np.ndarray = None) -> dict:
    """
    Full augmentation pipeline for a given stage.
    Works directly on Hypersim native resolution (768x1024).
    Returns dict with all outputs.
    """
    # Stage configurations for Hypersim 768x1024:
    # (divisor, sensor_h, sensor_w)
    # Sensor is 640x480 at full res, so at each divisor:
    # sensor = (480/divisor) x (640/divisor)  [H=480, W=640]
    # RGB target computed from downsampling 768x1024 by 1/divisor
    stages = {
        1: (8, 60, 80),    # ÷8: sensor 60x80
        2: (5, 96, 128),   # ÷5: sensor 96x128
        3: (4, 120, 160),  # ÷4: sensor 120x160
        4: (2, 240, 320),  # ÷2: sensor 240x320
    }

    if stage not in stages:
        raise ValueError(f"Unknown stage {stage}")

    divisor, sensor_h, sensor_w = stages[stage]
    scale = 1.0 / divisor

    # 1. Downsample RGB and GT depth to target RGB
    #for this stage
    rgb_ds = downsample_image(rgb, scale, cv2.INTER_AREA)
    gt_depth_ds = downsample_image(gt_depth, scale, cv2.INTER_NEAREST)

    # Also downsample euclidean distance if provided
    if euclidean_distance is not None:
        euclidean_ds = downsample_image(euclidean_distance, scale, cv2.INTER_NEAREST)
    else:
        euclidean_ds = None

    # Apply depth range mask to GT depth
    gt_valid_mask = apply_depth_range_mask(gt_depth_ds)
    gt_depth_ds = gt_depth_ds.copy()
    gt_depth_ds[~gt_valid_mask] = 0.0

    # Get actual downsampled RGB dimensions
    rgb_h, rgb_w = rgb_ds.shape[:2]

    # 2. Random crop GT depth to sensor resolution
    gt_cropped, perimeter = random_crop(gt_depth_ds, sensor_h, sensor_w, rng)
    # Also crop the GT valid mask
    gt_valid_cropped, _ = random_crop(gt_valid_mask.astype(np.float32), sensor_h, sensor_w, rng)
    gt_valid_cropped = gt_valid_cropped > 0.5
    perimeter.scale = 1.0  # Crop coords already in stage RGB space; reprojection target is same space

    # 3. Sparsify cropped depth + add noise (sparsify_depth already applies depth range)
    sparse_depth, valid_mask = sparsify_depth(gt_cropped, sparsity, rng, noise_std)

    # 4. Reproject to high-res RGB plane using nearest neighbor (with extrinsic calibration error)
    reprojected_depth, reprojected_mask = reproject_nearest(
        sparse_depth, valid_mask, perimeter, rgb_h, rgb_w,
        rng=rng,
        extrinsics_trans_px=extrinsics_trans_px,
        extrinsics_rot_deg=extrinsics_rot_deg
    )

    # Apply depth range mask to reprojected depth as well
    reprojected_valid = apply_depth_range_mask(reprojected_depth)
    reprojected_mask = (reprojected_mask > 0.5) & reprojected_valid
    reprojected_depth = reprojected_depth.copy()
    reprojected_depth[~reprojected_valid] = 0.0

    # 5. Also downsample RGB to sensor resolution for reference
    rgb_sensor = downsample_image(rgb_ds, sensor_h / rgb_h, cv2.INTER_AREA)

    return {
        "rgb": rgb_ds,                    # High-res RGB (target resolution for stage)
        "gt_depth": gt_depth_ds,          # High-res GT depth (target resolution for stage)
        "gt_valid_mask": gt_valid_mask,   # High-res GT validity mask
        "sparse_depth": sparse_depth,     # Low-res sparse depth (sensor resolution)
        "sparse_valid_mask": valid_mask,  # Low-res validity mask
        "reprojected_depth": reprojected_depth,  # High-res reprojected depth
        "reprojected_mask": reprojected_mask,    # High-res reprojection validity
        "rgb_sensor": rgb_sensor,         # RGB at sensor resolution
        "crop_perimeter": perimeter,      # Crop info for reference
        "stage": stage,
        "sparsity": sparsity,
        "euclidean_distance": euclidean_distance,  # Original Euclidean distance for error map
    }


def print_sparsity_stats(sparse_depth: np.ndarray, valid_mask: np.ndarray, name: str = ""):
    """Print statistics about sparsification."""
    valid_pixels = valid_mask.sum()
    total_pixels = valid_mask.size
    sparsity_pct = 100 * valid_pixels / total_pixels
    print(f"  {name}: {valid_pixels}/{total_pixels} valid ({sparsity_pct:.2f}%)")
    if valid_pixels > 0:
        valid_depths = sparse_depth[valid_mask > 0]
        print(f"    Depth range: [{valid_depths.min():.3f}, {valid_depths.max():.3f}], mean={valid_depths.mean():.3f}")


if __name__ == "__main__":
    # Quick self-test with synthetic data
    rng = np.random.default_rng(42)
    rgb = rng.random((768, 1024, 3)).astype(np.float32)
    gt_depth = rng.uniform(1, 5, (768, 1024)).astype(np.float32)

    for stage in [1, 2, 3, 4]:
        for sparsity in [0.01, 0.02, 0.05, 0.1, 0.2]:
            out = apply_augmentation_pipeline(rgb, gt_depth, stage, sparsity, 0.01, rng)
            print(f"Stage {stage}, sparsity={sparsity}: "
                  f"rgb={out['rgb'].shape}, gt={out['gt_depth'].shape}, "
                  f"sparse={out['sparse_depth'].shape}, reproj={out['reprojected_depth'].shape}")
            print_sparsity_stats(out['sparse_depth'], out['sparse_valid_mask'], "  sparse")
            print_sparsity_stats(out['reprojected_depth'], out['reprojected_mask'], "  reproj")
