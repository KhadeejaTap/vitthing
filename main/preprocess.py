#!/usr/bin/env python3
"""
Preprocessing: load Hypersim data, downsample and center-crop RGB + GT depth.
Converts euclidean distance to z-depth before downsampling.
Saves to hypersim_data/ with stage information in filename.
"""

import sys
import time
import cv2
import numpy as np
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from main.hypersim_dataset import (
    build_hypersim_index,
    load_sample,
)

# Stage target resolutions (H, W) with hardcoded sensor crop specs.
# Crop spec fields:
#   sensor_h, sensor_w, minx, maxx, miny, maxy, dx_min, dx_max, dy_min, dy_max
STAGE_CONFIGS = {
    1: {
        "rgb": (144, 256),  # 256x144 (W x H)
        "sensor_options": {
            "160x128": {
                "sensor_h": 128,
                "sensor_w": 160,
                "minx": 48,
                "maxx": 208,
                "miny": 8,
                "maxy": 136,
                "dx_min": -35,
                "dx_max": 35,
                "dy_min": -6,
                "dy_max": 6,
            },
            "176x128": {
                "sensor_h": 128,
                "sensor_w": 176,
                "minx": 40,
                "maxx": 216,
                "miny": 8,
                "maxy": 136,
                "dx_min": -30,
                "dx_max": 30,
                "dy_min": -6,
                "dy_max": 6,
            },
        },
    },
    2: {
        "rgb": (288, 512),  # 512x288 (W x H)
        "sensor_options": {
            "320x256": {
                "sensor_h": 256,
                "sensor_w": 320,
                "minx": 96,
                "maxx": 416,
                "miny": 16,
                "maxy": 272,
                "dx_min": -76,
                "dx_max": 76,
                "dy_min": -8,
                "dy_max": 8,
            },
            "352x256": {
                "sensor_h": 256,
                "sensor_w": 352,
                "minx": 80,
                "maxx": 432,
                "miny": 16,
                "maxy": 272,
                "dx_min": -66,
                "dx_max": 66,
                "dy_min": -8,
                "dy_max": 8,
            },
        },
    },
}

# Hypersim native resolution
HYPERSIM_H, HYPERSIM_W = 768, 1024
HYPERSIM_FOCAL = 886.81

# Precompute image plane for distance-to-depth conversion
_imageplane_x = np.linspace((-0.5 * HYPERSIM_W) + 0.5, (0.5 * HYPERSIM_W) - 0.5, HYPERSIM_W).reshape(1, HYPERSIM_W).repeat(HYPERSIM_H, 0).astype(np.float32)[:, :, None]
_imageplane_y = np.linspace((-0.5 * HYPERSIM_H) + 0.5, (0.5 * HYPERSIM_H) - 0.5, HYPERSIM_H).reshape(HYPERSIM_H, 1).repeat(HYPERSIM_W, 1).astype(np.float32)[:, :, None]
_imageplane_z = np.full([HYPERSIM_H, HYPERSIM_W, 1], HYPERSIM_FOCAL, np.float32)
_IMAGEPLANE = np.concatenate([_imageplane_x, _imageplane_y, _imageplane_z], 2)
_IMAGEPLANE_NORM = np.linalg.norm(_IMAGEPLANE, 2, 2)


def distance_to_depth(distance: np.ndarray) -> np.ndarray:
    """Convert Euclidean distance to planar depth (z-depth)."""
    return distance / _IMAGEPLANE_NORM * HYPERSIM_FOCAL


def compute_scale_and_crop(target_h, target_w, src_h=HYPERSIM_H, src_w=HYPERSIM_W):
    """
    Compute uniform scale so shorter dimension covers target, then center-crop excess.
    Returns: scale, crop_top, crop_left, crop_h, crop_w
    """
    scale_h = target_h / src_h
    scale_w = target_w / src_w
    scale = max(scale_h, scale_w)  # uniform scale so shorter dim covers target

    scaled_h = int(round(src_h * scale))
    scaled_w = int(round(src_w * scale))

    # Center-crop excess
    crop_top = (scaled_h - target_h) // 2
    crop_left = (scaled_w - target_w) // 2

    return scale, crop_top, crop_left, target_h, target_w


def downsample_and_crop(image, scale, crop_top, crop_left, crop_h, crop_w, interp=cv2.INTER_AREA):
    """Downsample by scale, then center-crop."""
    h, w = image.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    cropped = resized[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
    return cropped


def tonemap_rgb(rgb_image):
    """
    Apply Reinhard-style global tonemapping to RGB image.
    Converts linear RGB to tonemapped RGB suitable for display.
    """
    # Compute brightness using CCIR601 YIQ method (luminance)
    brightness = 0.3 * rgb_image[:, :, 0] + 0.59 * rgb_image[:, :, 1] + 0.11 * rgb_image[:, :, 2]

    # Tonemapping parameters
    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8

    # Create valid mask (non-zero brightness pixels)
    valid_mask = brightness > 0

    if np.count_nonzero(valid_mask) == 0:
        # If no valid pixels, return original image
        scale = 1.0
    else:
        brightness_valid = brightness[valid_mask]

        # Avoid division by zero
        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            # Reinhard-style tonemapping with gamma correction
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    # Apply tonemapping: scale then gamma compress
    rgb_tonemapped = np.power(np.maximum(scale * rgb_image, 0), gamma)

    return rgb_tonemapped


def process_and_save(samples, split, out_root, stage, rng, total_samples=None, start_index=0, simple=False):
    """Process all samples and save to disk in flat directory structure."""
    stage_start_time = time.time()
    stage_cfg = STAGE_CONFIGS[stage]
    target_h, target_w = stage_cfg["rgb"]
    sensor_option_names = tuple(stage_cfg["sensor_options"].keys())
    scale, crop_top, crop_left, crop_h, crop_w = compute_scale_and_crop(target_h, target_w)

    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Processing {len(samples)} samples for {split}...")
    print(f"    GT: {target_h}x{target_w}, random sensor option from {sensor_option_names}")
    for idx, sample in enumerate(samples):
        # Print progress every 50 samples
        if idx % 50 == 0:
            elapsed = time.time() - stage_start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"    Processed {idx}/{len(samples)} samples... ({rate:.1f} samples/sec)")

        sensor_option = rng.choice(sensor_option_names)
        sensor_cfg = stage_cfg["sensor_options"][sensor_option]
        sensor_h = sensor_cfg["sensor_h"]
        sensor_w = sensor_cfg["sensor_w"]
        minx = sensor_cfg["minx"]
        maxx = sensor_cfg["maxx"]
        miny = sensor_cfg["miny"]
        maxy = sensor_cfg["maxy"]
        dx_min = sensor_cfg["dx_min"]
        dx_max = sensor_cfg["dx_max"]
        dy_min = sensor_cfg["dy_min"]
        dy_max = sensor_cfg["dy_max"]

        dx = int(rng.integers(dx_min, dx_max + 1))
        dy = int(rng.integers(dy_min, dy_max + 1))
        x0, x1 = minx + dx, maxx + dx
        y0, y1 = miny + dy, maxy + dy

        loaded = load_sample(sample["color_path"], sample["depth_path"])
        color_key = loaded["color_keys"][0]
        depth_key = loaded["depth_keys"][0]

        # Load RGB (linear scene radiance)
        rgb = loaded["color_data"][color_key].astype(np.float32)

        # Apply tonemapping first (convert linear RGB to display-appropriate RGB)
        rgb = tonemap_rgb(rgb)

        # Get euclidean distance and convert to z-depth BEFORE downsampling
        euclidean_distance = loaded["euclidean_distance"]
        if euclidean_distance is not None:
            gt_depth = distance_to_depth(euclidean_distance.astype(np.float32))  # meters
        else:
            gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # meters

        # Downsample and crop
        rgb_ds = downsample_and_crop(rgb, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_AREA)
        gt_ds = downsample_and_crop(gt_depth, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_NEAREST)

        # Normalize RGB to [0,1] per image (handle exposure variations) AFTER cropping
        rgb_min = rgb_ds.min()
        rgb_max = rgb_ds.max()
        if rgb_max > rgb_min:
            rgb_ds = (rgb_ds - rgb_min) / (rgb_max - rgb_min)
        else:
            # Handle flat images
            rgb_ds = np.zeros_like(rgb_ds)

        # Scale to [0,255] for ImageNet normalization
        rgb_ds = rgb_ds * 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_ds = (rgb_ds / 255.0 - mean) / std

        # Get euclidean distance and convert to z-depth BEFORE downsampling
        euclidean_distance = loaded["euclidean_distance"]
        if euclidean_distance is not None:
            gt_depth = distance_to_depth(euclidean_distance.astype(np.float32))  # meters
        else:
            gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # meters

        # Downsample and crop
        rgb_ds = downsample_and_crop(rgb, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_AREA)
        gt_ds = downsample_and_crop(gt_depth, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_NEAREST)

        # Convert GT depth to mm
        gt_ds_mm = gt_ds * 1000.0

        # Ground truth mask: valid if >= 300mm and <= 20000mm (increased max range)
        # This naturally handles NaN values (NaN >= 300 is False, NaN <= 20000 is False)
        gt_mask = (gt_ds_mm >= 300.0) & (gt_ds_mm <= 20000.0)

        # Sensor depth: crop GT depth using chosen option and random valid shift.
        sensor_depth = gt_ds_mm.copy()
        sensor_mask = gt_mask.copy()

        if not simple:
            sensor_depth = sensor_depth[y0:y1, x0:x1]
            sensor_mask = sensor_mask[y0:y1, x0:x1]
            sh, sw = sensor_h, sensor_w
        else:
            # simple mode: no crop, sparsify straight from full GT depth
            sh, sw = target_h, target_w

        # Two-stage sparsity: base uniform + aggressive edge-only falloff
        # base_density = fraction of pixels that are VALID (varied by curriculum)
        if total_samples is not None:
            # Curriculum learning: start dense, gradually increase sparsity
            progress = (start_index + idx) / total_samples  # 0.0 to 1.0
            if progress < 1.0/3.0:
                # First 1/3: 40-50% base density (even denser for very stable start)
                base_density = rng.uniform(0.40, 0.50)
            elif progress < 2.0/3.0:
                # Middle 1/3: 15-20% base density
                base_density = rng.uniform(0.25, 0.39)
            else:
                # Final 1/3: 9-12% base density (sparser)
                base_density = rng.uniform(0.09, 0.20)
        else:
            # Original behavior if total_samples not provided
            base_density = rng.uniform(0.14, 0.19)
        corner_density = 0.05  # 5% at corners
        edge_density = 0.10    # 10% at edges

        # Compute normalized distance from center (0=center, 1=corners)
        cy, cx = sh / 2.0, sw / 2.0
        yy, xx = np.meshgrid(np.arange(sh), np.arange(sw), indexing='ij')
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        norm_dist = dist / max_dist

        # Very sharp falloff: only outer ~15% radius affected
        falloff_power = 12.0  # extremely sharp - center/mid untouched

        # Corner factor: only >0.85 radius gets hit
        corner_factor = np.clip((norm_dist - 0.85) / 0.15, 0, 1) ** falloff_power
        # Edge factor: only >0.7 radius
        edge_factor = np.clip((norm_dist - 0.7) / 0.3, 0, 1) ** 4.0

        # Density map: base in center/mid, drops only at edges/corners
        density_map = base_density * (1 - edge_factor * 0.5) * (1 - corner_factor * 0.8)
        density_map = np.where(corner_factor > 0.5, corner_density, density_map)
        density_map = np.where(edge_factor > 0.5, edge_density, density_map)
        density_map = np.clip(density_map, 0.02, 0.70)

        # Apply per-pixel: keep with probability = density
        sensor_mask = sensor_mask & (rng.random(sensor_mask.shape) < density_map)

        # Add Gaussian noise to valid sensor depth pixels (in mm)
        noise_std_mm = 5.0  # 5mm std
        valid_pixels = sensor_mask.astype(bool)
        if valid_pixels.any():
            noise = rng.normal(0, noise_std_mm, size=valid_pixels.sum())
            sensor_depth[valid_pixels] += noise

        if not simple:
            # Pad sensor depth and mask back to GT resolution using crop_perimeter
            # crop_perimeter = [y0, x0, sensor_h, sensor_w] in GT coordinates
            pad_top = y0
            pad_bottom = target_h - (y0 + sensor_h)
            pad_left = x0
            pad_right = target_w - (x0 + sensor_w)
            crop_perimeter = np.array([y0, x0, sensor_h, sensor_w], dtype=np.int32)
        else:
            # simple mode: sensor already at full GT res, padding is a no-op
            pad_top = pad_bottom = pad_left = pad_right = 0
            crop_perimeter = np.array([0, 0, target_h, target_w], dtype=np.int32)

        sensor_depth_padded = np.pad(
            sensor_depth,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        sensor_mask_padded = np.pad(
            sensor_mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=False
        )

        # Nearest-neighbor flood fill on entire padded sensor depth
        # sensor_mask_padded unchanged - only original sparse points remain valid
        from scipy.ndimage import distance_transform_edt
        if sensor_mask_padded.any():
            invalid = ~sensor_mask_padded
            if invalid.any():
                indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
                sensor_depth_padded = sensor_depth_padded[tuple(indices)]

        # Save as NPZ (padded sensor depth/mask to match GT resolution)
        # Include stage in filename to avoid collisions between stage 1 and stage 2 processing
        fname = f"{sample['scene']}_{sample['cam']}_frame{sample['frame']}_stage{stage}.npz"
        np.savez_compressed(
            out_dir / fname,
            rgb=rgb_ds.astype(np.float32),      # (H, W, 3) in [0,1]
            gt_depth=gt_ds_mm.astype(np.float32),  # (H, W) mm
            gt_mask=gt_mask.astype(np.bool_),   # (H, W) bool
            sensor_depth=sensor_depth_padded.astype(np.float32),  # (H, W) mm padded
            sensor_mask=sensor_mask_padded.astype(np.bool_),      # (H, W) bool padded
            crop_perimeter=crop_perimeter,      # [y, x, h, w] in GT coords
            sensor_crop_bounds=np.array([x0, x1, y0, y1], dtype=np.int32),  # [minx, maxx, miny, maxy]
            sensor_base_bounds=np.array([minx, maxx, miny, maxy], dtype=np.int32),
            sensor_shift=np.array([dx, dy], dtype=np.int32),  # [dx, dy]
            sensor_shift_range=np.array([dx_min, dx_max, dy_min, dy_max], dtype=np.int32),
            sensor_option=np.array(sensor_option, dtype=object),
            meta=np.array([sample["scene"], sample["cam"], sample["frame"], stage], dtype=object),
        )

    # Final progress update
    stage_elapsed = time.time() - stage_start_time
    print(f"    Completed processing {len(samples)} samples for {split} in {stage_elapsed:.2f} seconds ({len(samples)/stage_elapsed:.1f} samples/sec)")


def main():
    scenes_root = _project_root / "../ml-hypersim/evermotion_dataset/scenes"
    out_root = _project_root / "hypersim_data"
    seed = 42
    rng = np.random.default_rng(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true", help="Use simple sensor simulation (no cropping, shifting, or flood fill)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: hypersim_data/ next to project root)")
    args = parser.parse_args()

    if args.outdir is not None:
        out_root = Path(args.outdir)

    overall_start_time = time.time()
    print("Starting preprocessing...")
    print("Building dataset index...")
    all_samples = build_hypersim_index(scenes_root)
    print(f"Total frames: {len(all_samples)}")
    print(f"Total scenes: {len(set(s['scene'] for s in all_samples))}")

    # Split samples into first half (stage 1) and second half (stage 2)
    mid_point = len(all_samples) // 2
    first_half = all_samples[:mid_point]
    second_half = all_samples[mid_point:]

    print(f"\n=== Splitting samples ===")
    print(f"First half ({len(first_half)} samples): processed with stage 1 configuration")
    print(f"Second half ({len(second_half)} samples): processed with stage 2 configuration")
    if args.simple:
        print("Using simple sensor simulation")

    # Process first half with stage 1
    print(f"\n=== Processing first half with stage 1 ===")
    process_and_save(first_half, "first_half", out_root, 1, rng, total_samples=len(first_half), start_index=0, simple=args.simple)

    # Process second half with stage 2
    print(f"\n=== Processing second half with stage 2 ===")
    process_and_save(second_half, "second_half", out_root, 2, rng, total_samples=len(second_half), start_index=len(first_half), simple=args.simple)

    overall_elapsed = time.time() - overall_start_time
    print(f"\nDone! Saved to {out_root}")
    print(f"Total processing time: {overall_elapsed:.2f} seconds ({overall_elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
