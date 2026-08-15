from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Calibration constants (shared by NumPy and PyTorch paths)
# ---------------------------------------------------------------------------

TOF_BASE_WIDTH, TOF_BASE_HEIGHT = 640, 480
RGB_BASE_WIDTH, RGB_BASE_HEIGHT = 1920, 1080

K_TOF_BASE_NP = np.array([
    [544.462653, 0, (TOF_BASE_WIDTH - 1) / 2.0],
    [0, 544.462653, (TOF_BASE_HEIGHT - 1) / 2.0],
    [0, 0, 1],
], dtype=np.float64)

K_RGB_BASE_NP = np.array([
    [910.799450, 0, (RGB_BASE_WIDTH - 1) / 2.0],
    [0, 910.799450, (RGB_BASE_HEIGHT - 1) / 2.0],
    [0, 0, 1],
], dtype=np.float64)

R_NP = np.array([
    [ 0.9998933213423421, -0.001586456582046135,  0.014519954906713411],
    [ 0.0014369365883950588, 0.99994589849845628,  0.010302198277837599],
    [-0.014535513345618034, -0.010280234998687087, 0.99984150524978266],
], dtype=np.float64)

T_NP = np.array([0.0020168806248490076, 0.070622275765060152, -0.013673635196481401], dtype=np.float64)

# ---------------------------------------------------------------------------
# Module-level torch tensors — loaded once, moved to whatever device the
# caller puts them on (typically GPU alongside the model).
# ---------------------------------------------------------------------------

K_TOF_BASE: Tensor = torch.from_numpy(K_TOF_BASE_NP).float()   # (3, 3)
K_RGB_BASE: Tensor = torch.from_numpy(K_RGB_BASE_NP).float()   # (3, 3)
R_TORCH:     Tensor = torch.from_numpy(R_NP).float()            # (3, 3)
T_TORCH:     Tensor = torch.from_numpy(T_NP).float()            # (3,)

# Depth range (mm) — kept in sync with data.py
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 8333.0


# ---------------------------------------------------------------------------
# NumPy path (legacy CLI tool)
# ---------------------------------------------------------------------------

def _scale_intrinsics_np(
    base_k: np.ndarray,
    base_width: int, base_height: int,
    width: int, height: int,
    jitter_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale intrinsics to target resolution, then jitter fx/fy/cx/cy only."""
    scale_x = width / base_width
    scale_y = height / base_height
    k = base_k.copy()

    k[0, 0] *= scale_x
    k[0, 2] *= scale_x
    k[1, 1] *= scale_y
    k[1, 2] *= scale_y

    if jitter_scale > 0:
        randn = rng.standard_normal if rng is not None else np.random.normal
        k[0, 0] *= 1.0 + randn(0, jitter_scale)
        k[1, 1] *= 1.0 + randn(0, jitter_scale)
        k[0, 2] *= 1.0 + randn(0, jitter_scale)
        k[1, 2] *= 1.0 + randn(0, jitter_scale)

    return k


def project_depth_ideal(
    depth_mm, valid_mask, jitter_scale=0.0, quantize_mm=0,
    rgb_width=RGB_BASE_WIDTH, rgb_height=RGB_BASE_HEIGHT,
):
    """NumPy reprojection (legacy, used by CLI)."""
    Z = depth_mm.astype(np.float64) / 1000.0
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    K_tof = _scale_intrinsics_np(K_TOF_BASE_NP, TOF_BASE_WIDTH, TOF_BASE_HEIGHT, w, h, jitter_scale)
    K_rgb = _scale_intrinsics_np(K_RGB_BASE_NP, RGB_BASE_WIDTH, RGB_BASE_HEIGHT, rgb_width, rgb_height, jitter_scale)

    internal_mask = (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM) & np.isfinite(Z)
    combined_mask = internal_mask & valid_mask.astype(bool)

    if not np.any(combined_mask):
        return np.zeros((rgb_height, rgb_width), dtype=np.float32)

    K_inv = np.linalg.inv(K_tof)
    pixels = np.stack((u[combined_mask], v[combined_mask], np.ones_like(u[combined_mask])), axis=0)
    points_tof = (K_inv @ pixels) * Z[combined_mask]

    points_rgb = (R_NP @ points_tof) + T_NP[:, np.newaxis]

    pixels_rgb_h = K_rgb @ points_rgb
    u_rgb = pixels_rgb_h[0] / pixels_rgb_h[2]
    v_rgb = pixels_rgb_h[1] / pixels_rgb_h[2]
    z_rgb = points_rgb[2]

    in_bounds = (
        (u_rgb >= 0) & (u_rgb < rgb_width) &
        (v_rgb >= 0) & (v_rgb < rgb_height) &
        (z_rgb > 0)
    )

    u_idx = np.clip(np.round(u_rgb[in_bounds]).astype(np.int32), 0, rgb_width - 1)
    v_idx = np.clip(np.round(v_rgb[in_bounds]).astype(np.int32), 0, rgb_height - 1)
    z_vals = z_rgb[in_bounds].astype(np.float32)

    depth_proj = np.full((rgb_height, rgb_width), np.inf, dtype=np.float32)
    flat_idx = v_idx * rgb_width + u_idx
    np.minimum.at(depth_proj.ravel(), flat_idx, z_vals)

    depth_mm_out = np.zeros((rgb_height, rgb_width), dtype=np.float32)
    valid_proj = np.isfinite(depth_proj)
    depth_mm_out[valid_proj] = depth_proj[valid_proj] * 1000.0

    if quantize_mm > 0:
        mask = depth_mm_out > 0
        depth_mm_out[mask] = np.round(depth_mm_out[mask] / quantize_mm) * quantize_mm

    return depth_mm_out.astype(np.uint16)


# ---------------------------------------------------------------------------
# PyTorch path (GPU-native, used by the training pipeline)
# ---------------------------------------------------------------------------

def _scale_intrinsics_torch(
    base_k: Tensor,
    base_width: int, base_height: int,
    width: int, height: int,
    jitter_scale: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Scale intrinsics to target resolution, then jitter fx/fy/cx/cy only.

    Returns a new (3, 3) tensor on the same device/dtype as *base_k*.
    """
    scale_x = width / base_width
    scale_y = height / base_height
    k = base_k.clone()

    k[0, 0] *= scale_x
    k[0, 2] *= scale_x
    k[1, 1] *= scale_y
    k[1, 2] *= scale_y

    if jitter_scale > 0:
        # Draw 4 independent perturbations for fx, fy, cx, cy
        noise = torch.randn(4, device=k.device, dtype=k.dtype, generator=generator) * jitter_scale
        k[0, 0] *= (1.0 + noise[0])
        k[1, 1] *= (1.0 + noise[1])
        k[0, 2] *= (1.0 + noise[2])
        k[1, 2] *= (1.0 + noise[3])

    return k


def project_depth_torch(
    depth_mm: Tensor,
    valid_mask: Tensor,
    jitter_scale: float = 0.0,
    seed: int = 0,
    rgb_width: int = RGB_BASE_WIDTH,
    rgb_height: int = RGB_BASE_HEIGHT,
) -> tuple[Tensor, Tensor]:
    """GPU-native ToF→RGB depth reprojection.

    Args:
        depth_mm:   (1, H_tof, W_tof) float32 tensor in mm.
        valid_mask: (1, H_tof, W_tof) boolean or float tensor.
        jitter_scale: Std-dev of multiplicative jitter on intrinsics.
        seed:       Per-epoch seed for deterministic jitter.
                    Use  base_seed + (epoch // 10)  for the curriculum schedule.
        rgb_width / rgb_height: Output resolution.

    Returns:
        Tuple of:
            depth_mm_out: (1, H_rgb, W_rgb) float32 tensor in mm, invalid pixels are 0.
            valid_out:    (1, H_rgb, W_rgb) float32 validity mask (1 = valid, 0 = invalid).
    """
    device = depth_mm.device
    dtype  = torch.float32

    # Move calibration to the same device as the input
    K_tof_base = K_TOF_BASE.to(device=device, dtype=dtype)
    K_rgb_base = K_RGB_BASE.to(device=device, dtype=dtype)
    R = R_TORCH.to(device=device, dtype=dtype)
    T = T_TORCH.to(device=device, dtype=dtype)

    # Deterministic generator for this seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # Scale (and optionally jitter) intrinsics
    h_tof, w_tof = depth_mm.shape[-2:]
    K_tof = _scale_intrinsics_torch(K_tof_base, TOF_BASE_WIDTH, TOF_BASE_HEIGHT, w_tof, h_tof, jitter_scale, generator)
    K_rgb = _scale_intrinsics_torch(K_rgb_base, RGB_BASE_WIDTH, RGB_BASE_HEIGHT, rgb_width, rgb_height, jitter_scale, generator)

    # Depth in metres
    Z = depth_mm.float() / 1000.0

    # Validity mask
    mask_bool = (
        (depth_mm >= DEPTH_MIN_MM) &
        (depth_mm <= DEPTH_MAX_MM) &
        torch.isfinite(Z) &
        (valid_mask > 0)
    )  # (1, H, W)

    if not mask_bool.any():
        empty_depth = torch.zeros(1, rgb_height, rgb_width, device=device, dtype=dtype)
        empty_valid = torch.zeros(1, rgb_height, rgb_width, device=device, dtype=dtype)
        return empty_depth, empty_valid

    # Pixel grid  (H, W)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(h_tof, device=device, dtype=dtype),
        torch.arange(w_tof, device=device, dtype=dtype),
        indexing="ij",
    )

    # Flatten and mask — only process valid pixels
    # Squeeze the batch dim so mask is (H, W) and depth is (H, W)
    mask_2d = mask_bool.squeeze(0)   # (H, W)
    Z_2d   = Z.squeeze(0)           # (H, W)

    u_flat = u_grid[mask_2d]         # (N,)
    v_flat = v_grid[mask_2d]         # (N,)
    z_flat = Z_2d[mask_2d]          # (N,)

    ones = torch.ones_like(u_flat)          # (N,)
    pixels = torch.stack([u_flat, v_flat, ones], dim=0)  # (3, N)

    # Back-project to 3D in ToF camera frame
    K_tof_inv = torch.linalg.inv(K_tof)
    points_tof = (K_tof_inv @ pixels) * z_flat.unsqueeze(0)  # (3, N)

    # Rigid transform to RGB camera frame
    points_rgb = (R @ points_tof) + T.unsqueeze(1)  # (3, N)

    # Project into RGB pixel coordinates
    pixels_rgb_h = K_rgb @ points_rgb  # (3, N)
    u_rgb = pixels_rgb_h[0] / pixels_rgb_h[2]
    v_rgb = pixels_rgb_h[1] / pixels_rgb_h[2]
    z_rgb = points_rgb[2]

    # In-bounds filter
    in_bounds = (
        (u_rgb >= 0) & (u_rgb < rgb_width) &
        (v_rgb >= 0) & (v_rgb < rgb_height) &
        (z_rgb > 0)
    )

    u_idx = torch.clamp(torch.round(u_rgb[in_bounds]).long(), 0, rgb_width - 1)
    v_idx = torch.clamp(torch.round(v_rgb[in_bounds]).long(), 0, rgb_height - 1)
    z_vals = z_rgb[in_bounds]

    # Z-buffer via scatter_reduce("amin")
    depth_proj = torch.full((rgb_height * rgb_width,), float("inf"), device=device, dtype=dtype)
    flat_idx = v_idx * rgb_width + u_idx  # (M,)

    # scatter_reduce: 1D source → 1D output, takes minimum depth per pixel
    depth_proj = depth_proj.scatter_reduce(0, flat_idx, z_vals, reduce="amin", include_self=True)
    depth_proj = depth_proj.reshape(rgb_height, rgb_width)

    # Convert metres → mm and zero out unfilled pixels
    valid_proj = torch.isfinite(depth_proj)
    depth_mm_out = torch.where(
        valid_proj,
        depth_proj * 1000.0,
        torch.zeros_like(depth_proj),
    )

    # Validity mask: 1.0 where a valid depth was reprojected, 0.0 elsewhere
    valid_out = valid_proj.float()

    return depth_mm_out.unsqueeze(0), valid_out.unsqueeze(0)  # (1, H_rgb, W_rgb) each


# ---------------------------------------------------------------------------
# CLI (still uses the NumPy path for offline batch processing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--jitter-scale", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=RGB_BASE_WIDTH)
    parser.add_argument("--height", type=int, default=RGB_BASE_HEIGHT)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(input_dir.glob("frame_*_tof_mm.npy"))
    print(f"Reprojection (jitter={args.jitter_scale}, res={args.width}x{args.height}): Processing {len(depth_files)} frames...")

    for fpath in depth_files:
        frame_num = fpath.stem.split("_")[1]
        depth_mm = np.load(fpath)

        # Look for the mask relative to the depth file
        mask_path = input_dir.parent / "processed" / f"frame_{frame_num}_valid_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing validity mask: {mask_path}")

        valid_mask = np.load(mask_path)

        proj_mm = project_depth_ideal(depth_mm, valid_mask, args.jitter_scale, rgb_width=args.width, rgb_height=args.height)
        np.save(out_dir / f"frame_{frame_num}_depth_proj_mm.npy", proj_mm)
    print("Done!")
