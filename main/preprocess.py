#!/usr/bin/env python3
"""
Preprocessing: load Hypersim data, split into stages, downsample and center-crop RGB + GT depth.
Converts euclidean distance to z-depth before downsampling.
Saves to hypersim_data/stage{N}/{split}/
"""

import sys
import cv2
import numpy as np
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from main.hypersim_dataset import (
    build_hypersim_index,
    split_by_frame,
    split_train_stages,
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
    print(f"[DEBUG] After downsampling (before crop): resized shape: {resized.shape}")
    cropped = resized[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
    return cropped


def process_and_save(samples, split, out_root, stage, rng):
    """Process all samples for a stage/split and save to disk."""
    stage_cfg = STAGE_CONFIGS[stage]
    target_h, target_w = stage_cfg["rgb"]
    sensor_option_names = tuple(stage_cfg["sensor_options"].keys())
    scale, crop_top, crop_left, crop_h, crop_w = compute_scale_and_crop(target_h, target_w)

    out_dir = out_root / f"stage{stage}" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Processing {len(samples)} samples for stage {stage} {split}...")
    print(f"    GT: {target_h}x{target_w}, random sensor option from {sensor_option_names}")
    for sample in samples:
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

        rgb = loaded["color_data"][color_key].astype(np.float32)  # (H, W, 3) raw HDF5 values

        # GET RENDER ENTITY ID FOR TONEMAPPING
        if "render_entity_id" in loaded["color_data"]:
            render_entity_id = loaded["color_data"]["render_entity_id"].astype(np.int32)
        else:
            # Fallback: if not available, use all pixels as valid
            render_entity_id = np.ones(rgb.shape[:2], dtype=np.int32)  # All valid (shape H,W)

        # APPLY TONEMAPPING
        gamma = 1.0 / 2.2   # standard gamma correction exponent
        inv_gamma = 1.0 / gamma
        percentile = 90        # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling

        valid_mask = render_entity_id != -1

        if np.count_nonzero(valid_mask) == 0:
            scale = 1.0 # if there are no valid pixels, then set scale to 1.0
        else:
            brightness = 0.3*rgb[:,:,0] + 0.59*rgb[:,:,1] + 0.11*rgb[:,:,2] # "CCIR601 YIQ" method for computing brightness
            brightness_valid = brightness[valid_mask]

            eps = 0.0001 # if the nth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
            brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

            if brightness_nth_percentile_current < eps:
                scale = 0.0
            else:
                # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
                # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
                #
                # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
                # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

        rgb = np.power(np.maximum(scale*rgb, 0), gamma)
        # Clip to [0,1] to ensure valid range for model input
        rgb_tonemapped = np.clip(rgb, 0.0, 1.0)


        # Note: We will delete the original H5 file after all processing is complete

        # PROCESS DEPTH (unchanged from original preprocessing)
        gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # (H, W) planar z-depth in meters
        euclidean_distance = loaded.get("euclidean_distance", None)
        if euclidean_distance is not None:
            gt_depth = distance_to_depth(euclidean_distance.astype(np.float32))  # meters
        else:
            gt_depth = loaded["depth_data"][depth_key].astype(np.float32)  # meters

        # Downsample and crop - COMPUTE SCALE PER SAMPLE BASED ON ACTUAL IMAGE DIMENSIONS
        scale, crop_top, crop_left, crop_h, crop_w = compute_scale_and_crop(target_h, target_w,
                                                                          rgb_tonemapped.shape[0],
                                                                          rgb_tonemapped.shape[1])
        rgb_ds = downsample_and_crop(rgb_tonemapped, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_AREA)
        gt_ds = downsample_and_crop(gt_depth, scale, crop_top, crop_left, crop_h, crop_w, cv2.INTER_NEAREST)
        print(f"[DEBUG] After downsample and crop: rgb_ds shape: {rgb_ds.shape}, gt_ds shape: {gt_ds.shape}")

        # Convert GT depth to mm
        gt_ds_mm = gt_ds * 1000.0

        # Ground truth mask: valid if within sensor range [300, 8333] mm
        gt_mask = (gt_ds_mm >= 300.0) & (gt_ds_mm <= 8333.0)

        # Sensor depth: crop GT depth using chosen option and random valid shift.
        sensor_depth = gt_ds_mm.copy()
        sensor_mask = gt_mask.copy()

        sensor_depth = sensor_depth[y0:y1, x0:x1]
        sensor_mask = sensor_mask[y0:y1, x0:x1]

        # Two-stage sparsity: base uniform + aggressive edge-only falloff
        # base_density = fraction of pixels that are VALID (14-19%)
        base_density = rng.uniform(0.14, 0.19)
        corner_density = 0.05  # 5% at corners
        edge_density = 0.10    # 10% at edges

        # Compute normalized distance from center (0=center, 1=corners) based on actual cropped sensor mask dimensions
        sensor_h_actual, sensor_w_actual = sensor_mask.shape
        cy, cx = sensor_h_actual / 2.0, sensor_w_actual / 2.0
        yy, xx = np.meshgrid(np.arange(sensor_h_actual), np.arange(sensor_w_actual), indexing='ij')
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
        density_map = np.clip(density_map, 0.02, 0.25)

        # Apply per-pixel: keep with probability = density
        sensor_mask = sensor_mask & (rng.random(sensor_mask.shape) < density_map)

        # Add Gaussian noise to valid sensor depth pixels (in mm)
        noise_std_mm = 5.0  # 5mm std
        valid_pixels = sensor_mask.astype(bool)
        if valid_pixels.any():
            noise = rng.normal(0, noise_std_mm, size=valid_pixels.sum())
            sensor_depth[valid_pixels] += noise

        # Pad sensor depth and mask back to GT resolution using crop_perimeter
        # crop_perimeter = [y0, x0, sensor_h, sensor_w] in GT coordinates
        pad_top = y0
        pad_bottom = target_h - (y0 + sensor_h)
        pad_left = x0
        pad_right = target_w - (x0 + sensor_w)

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

        # Save crop perimeter (in GT coordinates)
        crop_perimeter = np.array([y0, x0, sensor_h, sensor_w], dtype=np.int32)
        # Prepare RGB for NPZ: transpose to (3, H, W) and apply ImageNet normalization
        rgb_npz = np.transpose(rgb_ds, (2, 0, 1))  # (3, H, W)
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb_npz = (rgb_npz - imagenet_mean) / imagenet_std

        # Save as NPZ (padded sensor depth/mask to match GT resolution)
        fname = f"{sample['scene']}_{sample['cam']}_frame{sample['frame']}.npz"
        np.savez_compressed(
            out_dir / fname,
            rgb=rgb_npz.astype(np.float32),      # (3, H, W) ImageNet-normalized
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


def main():
    scenes_root = os.environ.get("HYPERSIM_ROOT", str(Path.home() / "ml-hypersim" / "evermotion_dataset" / "scenes"))
    out_root = Path(__file__).resolve().parent.parent / "hypersim_data"
    val_fraction = 0.05
    seed = 42
    rng = np.random.default_rng(seed)

    print("Building dataset index...")
    all_samples = build_hypersim_index(scenes_root)
    print(f"Total frames: {len(all_samples)}")
    print(f"Total scenes: {len(set(s['scene'] for s in all_samples))}")

    # Debug: Show first few samples
    if len(all_samples) > 0:
        print(f"First sample: {all_samples[0]}")
        if len(all_samples) > 1:
            print(f"Last sample: {all_samples[-1]}")

    train_samples, val_samples, n_val = split_by_frame(all_samples, val_fraction)
    print(f"\nTrain: {len(train_samples)} frames")
    print(f"Val:   {len(val_samples)} frames ({n_val})")

    stage_samples = split_train_stages(train_samples)
    print("\n--- Train split by stage ---")
    for i, samples in enumerate(stage_samples):
        print(f"  Stage {i+1}: {len(samples)} frames")

    # Process and save all stages
    for stage in [1, 2]:
        print(f"\n=== Stage {stage} ===")
        print(f"Processing {len(stage_samples[stage - 1])} train samples")
        process_and_save(stage_samples[stage - 1], "train", out_root, stage, rng)
        print(f"Processing {len(val_samples)} val samples")
        process_and_save(val_samples, "val", out_root, stage, rng)

    print(f"\nDone! Saved to {out_root}")


if __name__ == "__main__":
    main()
