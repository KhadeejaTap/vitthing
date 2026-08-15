#!/usr/bin/env python3
"""
Visualizer for preprocessed Hypersim NPZ outputs.
Shows each saved field individually.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob


def _extract_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _depth_range_mm(depth_mm: np.ndarray, mask: np.ndarray = None):
    if mask is not None:
        valid = depth_mm[mask & np.isfinite(depth_mm) & (depth_mm > 0)]
    else:
        valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
    if valid.size == 0:
        return None, None
    return np.percentile(valid, 2), np.percentile(valid, 98)


def visualize_one(npz_path: Path, out_dir: Path):
    with np.load(npz_path, allow_pickle=True) as data:
        rgb = data.get("rgb")
        gt_depth = data.get("gt_depth")
        gt_mask = data.get("gt_mask")
        sensor_depth = data.get("sensor_depth")
        sensor_mask = data.get("sensor_mask")
        crop_perimeter = data.get("crop_perimeter")
        sensor_crop_bounds = data.get("sensor_crop_bounds")
        sensor_base_bounds = data.get("sensor_base_bounds")
        sensor_shift = data.get("sensor_shift")
        sensor_shift_range = data.get("sensor_shift_range")
        sensor_option = _extract_scalar(data.get("sensor_option"))
        meta = data.get("meta")

    stem = npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) RGB
    if rgb is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(np.clip(rgb, 0, 1))
        shift_str = f"[{sensor_shift[0]}, {sensor_shift[1]}]" if sensor_shift is not None else "None"
        ax.set_title(f"{stem}\noption={sensor_option} shift={shift_str}")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_rgb.png", dpi=150)
        plt.close(fig)

    # 2) GT depth
    if gt_depth is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        gt = gt_depth.astype(np.float32)
        if gt_mask is not None:
            gt = np.where(gt_mask.astype(bool), gt, np.nan)
        vmin, vmax = _depth_range_mm(gt, gt_mask)
        im = ax.imshow(gt, cmap="magma", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label="depth (mm)")
        ax.set_title("GT depth")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_gt_depth.png", dpi=150)
        plt.close(fig)

    # 3) GT mask
    if gt_mask is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(gt_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"GT mask (valid={gt_mask.mean()*100:.1f}%)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_gt_mask.png", dpi=150)
        plt.close(fig)

    # 4) Sensor depth (padded, invalid=0)
    if sensor_depth is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        sd = sensor_depth.astype(np.float32)
        vmin, vmax = _depth_range_mm(sd, sensor_mask)
        im = ax.imshow(sd, cmap="magma", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label="depth (mm)")
        ax.set_title(f"Sensor depth (padded, valid={sensor_mask.mean()*100:.1f}%)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_sensor_depth.png", dpi=150)
        plt.close(fig)

    # 5) Sensor mask
    if sensor_mask is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(sensor_mask.astype(float), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Sensor mask (valid={sensor_mask.mean()*100:.1f}%)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_sensor_mask.png", dpi=150)
        plt.close(fig)

    # 6) Sensor depth crop only (zoomed to crop region)
    if sensor_depth is not None and crop_perimeter is not None:
        cy, cx, ch, cw = [int(x) for x in crop_perimeter]
        crop = sensor_depth[cy:cy+ch, cx:cx+cw]
        crop_mask = sensor_mask[cy:cy+ch, cx:cx+cw] if sensor_mask is not None else None
        fig, ax = plt.subplots(figsize=(6, 4))
        c = crop.astype(np.float32)
        if crop_mask is not None:
            c = np.where(crop_mask.astype(bool), c, np.nan)
        vmin, vmax = _depth_range_mm(c, crop_mask)
        im = ax.imshow(c, cmap="magma", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label="depth (mm)")
        ax.set_title(f"Sensor depth crop only ({ch}x{cw})")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_sensor_depth_crop.png", dpi=150)
        plt.close(fig)

    # 7) Print metadata
    print(f"\n{stem}:")
    print(f"  meta: {meta}")
    print(f"  sensor_option: {sensor_option}")
    print(f"  sensor_shift: {sensor_shift}")
    print(f"  sensor_shift_range: {sensor_shift_range}")
    print(f"  crop_perimeter: {crop_perimeter}")
    print(f"  sensor_crop_bounds: {sensor_crop_bounds}")
    print(f"  sensor_base_bounds: {sensor_base_bounds}")
    if rgb is not None:
        print(f"  rgb: {rgb.shape} [{rgb.min():.3f}, {rgb.max():.3f}]")
    if gt_depth is not None:
        print(f"  gt_depth: {gt_depth.shape} [{gt_depth[gt_mask].min():.1f}, {gt_depth[gt_mask].max():.1f}] mm")
    if sensor_depth is not None and sensor_mask is not None:
        valid = sensor_depth[sensor_mask]
        print(f"  sensor_depth: {sensor_depth.shape} valid={valid.size} [{valid.min():.1f}, {valid.max():.1f}] mm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None, help="Path to hypersim_data root")
    p.add_argument("--max", type=int, default=0, help="Max number of files to render (0 = all)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle files before rendering")
    args = p.parse_args()

    if args.path is None:
        root = (Path(__file__).resolve().parent.parent / "hypersim_data").resolve()
    else:
        root = Path(args.path).resolve()

    files = sorted(glob.glob(str(root / "**" / "*.npz"), recursive=True))
    if len(files) == 0:
        print("No npz files found under", root)
        return

    if args.shuffle:
        rng = np.random.default_rng(0)
        rng.shuffle(files)

    limit = len(files) if args.max <= 0 else min(args.max, len(files))
    files = files[:limit]
    print(f"Rendering {len(files)} files from {root}")

    for idx, f in enumerate(files, start=1):
        npz_path = Path(f)
        out_dir = npz_path.parent / "vis" / npz_path.stem
        visualize_one(npz_path, out_dir)
        if idx % 10 == 0 or idx == len(files):
            print(f"  saved {idx}/{len(files)}")


if __name__ == "__main__":
    main()
