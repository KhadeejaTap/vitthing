#!/usr/bin/env python3
"""
Overfit a single sample for N iterations.

Sanity check: if the model can't drive the loss to ~0 on one sample,
something is broken (loss bug, detached gradients, bad init, etc.).

Usage:
    python code/overfit_one.py --iters 500 --stage 1
    python code/overfit_one.py --iters 1000 --stage 1 --init random --lr 1e-3
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from encoder import load_backbone, pad_to_multiple, crop_to_original
from fusion import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params
from losses import total_loss
from hypersim_dataset import HypersimDataset, build_hypersim_index, split_by_frame


def parse_args():
    p = argparse.ArgumentParser(description="Overfit one sample for N iters")
    p.add_argument("--scenes-root", type=str,
                   default="/home/khadeeja/ml-hypersim/evermotion_dataset/scenes")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Index into the train split to overfit (default: 0)")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                   help="Dataset stage (resolution). Default 1 (smallest, fastest).")
    p.add_argument("--iters", type=int, default=500, help="Number of gradient steps")
    p.add_argument("--init", type=str, default="pretrained",
                   choices=["pretrained", "random", "shared_random"],
                   help="Backbone init (default: pretrained)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for ALL params (default: 1e-4)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Weight decay (default: 0 — we want pure overfit)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # Loss kwargs (match train_staged defaults)
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--radius-2d", type=float, default=0.05)
    p.add_argument("--min-points", type=int, default=16)
    p.add_argument("--local-downsample", type=int, default=4)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--vis-every", type=int, default=10,
                   help="Save a visualization every N iters (default: 10)")
    p.add_argument("--vis-dir", type=str, default="overfit_vis",
                   help="Directory for visualization PNGs (default: overfit_vis)")
    return p.parse_args()


def compute_intrinsics(stage, height, width):
    """Same scaling as train_staged.compute_intrinsics."""
    scale = {1: 1.0 / 8.0, 2: 1.0 / 5.0, 3: 1.0 / 4.0, 4: 1.0 / 2.0}[stage]
    focal = 886.81
    fx = focal * scale
    fy = focal * scale
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


# Turbo colormap (matches the depth-viz convention in the field)
_TURBO = LinearSegmentedColormap.from_list(
    "turbo_approx",
    ["#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4",
     "#24eca6", "#61fc6c", "#a4fc3b", "#d1e834", "#f6c438",
     "#fb8b32", "#f65322", "#cb1b1b", "#7d0a4e", "#30123b"],
)


def _depth_to_color(d, vmin, vmax):
    """Normalize depth to [0,1] and apply colormap. d: (H,W) tensor -> (H,W,3) numpy."""
    d = d.detach().cpu().float()
    d = (d - vmin) / max(vmax - vmin, 1e-6)
    d = d.clamp(0, 1).numpy()
    return (_TURBO(d)[..., :3] * 255).astype(np.uint8)


def save_viz(path, rgb, depth_input, depth_pred, gt_depth, sensor_mask, gt_mask,
             alpha, beta, loss_val, it):
    """
    Save a 5-panel visualization: RGB | GT depth | Pred depth | Sensor input | |Pred-GT| error.
    All depth maps share the same color scale (computed from GT) for direct comparison.
    """
    # Squeeze batch + channel dims -> (H,W)
    rgb_img = rgb[0].detach().cpu().permute(1, 2, 0).numpy()           # (H,W,3) in [0,1]
    pred = depth_pred[0, 0].detach().cpu()                            # (H,W) mm
    gt = gt_depth[0, 0].detach().cpu()                                 # (H,W) mm
    smask = sensor_mask[0].detach().cpu()                              # (H,W)
    gmask = gt_mask[0, 0].detach().cpu()                               # (H,W)

    gt_valid = gmask > 0.5
    vmin = float(gt[gt_valid].min().item()) if gt_valid.any() else float(gt.min())
    vmax = float(gt[gt_valid].max().item()) if gt_valid.any() else float(gt.max())

    # Decode the ACTUAL sensor input from depth_input tensor.
    # depth_input ch0/ch1 = flood-filled log-normalized depth in [-1,1] (nearest-neighbor
    # filled across the whole frame by augment.reproject_nearest), ch2 = sparse mask in [-1,1]
    # (only ~11% real sensor points, NOT the filled pixels).
    # Show the flood-filled depth everywhere (what the model actually sees), then overlay
    # the sparse real-measurement mask as bright dots so the two are distinguishable.
    di = depth_input[0].detach().cpu()                                 # (3,H,W)
    zhat = (di[0] + 1.0) / 2.0                                         # (H,W) normalized depth
    sensor_depth = torch.exp(alpha * zhat + beta)                      # (H,W) mm, flood-filled everywhere
    sensor_color = _depth_to_color(sensor_depth, vmin, vmax)           # (H,W,3) uint8
    # Mark real sensor pixels with a bright yellow dot so they're visible on top of the fill
    real_px = (smask > 0.5).numpy()
    if real_px.any():
        sensor_color[real_px] = [255, 240, 0]  # bright yellow = real measurement

    # Error map (mm), only where GT valid
    err = (pred - gt).abs()
    err_img = torch.where(gt_valid, err, torch.zeros_like(err))
    err_max = float(err_img.max().item()) if err_img.max() > 0 else 1.0

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = [
        ("RGB", rgb_img.clip(0, 1), None),
        ("GT depth (mm)", _depth_to_color(gt, vmin, vmax), None),
        ("Pred depth (mm)", _depth_to_color(pred, vmin, vmax), None),
        ("Sensor input (yellow=real)", sensor_color, None),
        ("|Pred - GT| (mm)", err_img.numpy(), "magma"),
    ]
    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap is None and img.ndim == 3:
            ax.imshow(img)
        elif cmap is None:
            ax.imshow(img, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=err_max)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"iter {it} | loss={loss_val:.5f} | depth range [{vmin:.0f}, {vmax:.0f}] mm",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- Build dataset, grab one sample ----
    print(f"Building dataset index (stage {args.stage})...")
    all_samples = build_hypersim_index(args.scenes_root)
    train_samples, _, _ = split_by_frame(all_samples, val_fraction=0.1)
    print(f"Train split: {len(train_samples)} frames. Using sample idx={args.sample_idx}.")

    ds = HypersimDataset(
        args.scenes_root,
        stage=args.stage,
        split="train",
        seed=args.seed,
        samples=train_samples,
    )
    sample = ds[args.sample_idx]
    scene = f"{sample['scene']}/{sample['cam']}/frame.{sample['frame']}"
    print(f"Sample: {scene}")

    # Move to device, add batch dim
    rgb = sample["rgb"].unsqueeze(0).to(device)          # (1,3,H,W)
    depth_input = sample["depth_input"].unsqueeze(0).to(device)
    gt_depth = sample["gt_depth"].unsqueeze(0).to(device)  # (1,1,H,W) mm
    gt_mask = sample["gt_mask"].unsqueeze(0).to(device)
    sensor_mask = sample["valid_mask"].unsqueeze(0).to(device)

    # combined mask = GT ∩ sensor (same as train_staged)
    combined_mask = (gt_mask > 0.5) & sensor_mask.unsqueeze(1).bool()
    combined_mask = combined_mask.float()

    orig_h, orig_w = rgb.shape[-2], rgb.shape[-1]
    fx, fy, cx, cy = compute_intrinsics(args.stage, orig_h, orig_w)

    # Pad to patch multiple (frozen — same padded size every iter)
    rgb_p = pad_to_multiple(rgb)
    depth_input_p = pad_to_multiple(depth_input)
    padded_h, padded_w = rgb_p.shape[-2], rgb_p.shape[-1]
    h_patch, w_patch = padded_h // 14, padded_w // 14

    print(f"Input: rgb={rgb.shape} -> padded {rgb_p.shape}, patches {h_patch}x{w_patch}")

    # ---- Sensor data & ground truth: shapes + ranges ----
    sensor_valid = sensor_mask > 0.5
    gt_valid = gt_mask > 0.5

    print("\n--- sensor data ---")
    print(f"  depth_input: shape={tuple(depth_input.shape)}, "
          f"range=[{depth_input.min().item():.4f}, {depth_input.max().item():.4f}] "
          f"(expect ~[-1,1])")
    # depth_input ch2 is the validity mask in [-1,1] encoding; decode it
    di_mask = (depth_input[:, 2] + 1) / 2  # [-1,1] -> [0,1]
    print(f"  depth_input ch2 (mask decoded): "
          f"{int((di_mask > 0.5).sum().item())} valid px "
          f"({100*(di_mask > 0.5).float().mean().item():.2f}%)")
    print(f"  valid_mask (sensor): shape={tuple(sensor_mask.shape)}, "
          f"{int(sensor_valid.sum().item())} valid px "
          f"({100*sensor_valid.float().mean().item():.2f}%)")

    print("\n--- ground truth ---")
    print(f"  gt_depth: shape={tuple(gt_depth.shape)}, dtype={gt_depth.dtype}, "
          f"range=[{gt_depth[gt_valid].min().item():.1f}, "
          f"{gt_depth[gt_valid].max().item():.1f}] mm "
          f"(all px: [{gt_depth.min().item():.1f}, {gt_depth.max().item():.1f}])")
    print(f"  gt_mask: shape={tuple(gt_mask.shape)}, "
          f"{int(gt_valid.sum().item())} valid px "
          f"({100*gt_valid.float().mean().item():.1f}%)")
    print(f"  combined (GT∩sensor): {int(combined_mask.sum().item())} px "
          f"({100*combined_mask.mean().item():.2f}%)")

    # ---- Build model ----
    print(f"\nBuilding model (init={args.init})...")
    model_rgb = load_backbone(init_mode=args.init)
    model_depth = load_backbone(init_mode=args.init)
    encoder = DualBranchEncoder(model_rgb, model_depth).to(device)
    decoder = DPTDecoder().to(device)

    alpha, beta = compute_log_params()
    print(f"alpha={alpha:.4f} beta={beta:.4f}")

    # Single optimizer, single LR for everything — we want to overfit, not generalize
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    encoder.train()
    decoder.train()

    # ---- Overfit loop ----
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOverfitting for {args.iters} iters @ lr={args.lr} ...\n")
    t0 = time.time()
    best_loss = float("inf")
    first_loss = None

    for it in range(args.iters):
        optimizer.zero_grad()

        features = encoder(rgb_p, depth_input_p)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, padded_h, padded_w)

        depth_hat = crop_to_original(depth_hat, orig_h, orig_w)
        mask_logit = crop_to_original(mask_logit, orig_h, orig_w)
        depth_pred = denormalize_depth(depth_hat, alpha, beta)

        loss, parts = total_loss(
            depth_pred, gt_depth, combined_mask, mask_logit, combined_mask,
            fx, fy, cx, cy,
            level=args.level, radius_2d=args.radius_2d,
            min_points_per_patch=args.min_points,
            local_loss_downsample=args.local_downsample,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val
        best_loss = min(best_loss, loss_val)

        if it % args.print_every == 0 or it == args.iters - 1:
            # quick metric on pure GT mask
            with torch.no_grad():
                valid = gt_mask.bool()
                if valid.any():
                    pv = depth_pred[valid]
                    gv = gt_depth[valid]
                    absrel = ((pv - gv).abs() / gv.clamp_min(1e-6)).mean().item()
                    mae = (pv - gv).abs().mean().item()
                else:
                    absrel = float("nan")
                    mae = float("nan")

            dt = time.time() - t0
            print(f"  iter {it:4d}/{args.iters}  loss={loss_val:.5f}  "
                  f"absrel={absrel:.4f}  mae={mae:.1f}mm  ({dt:.1f}s)")

        # Visualization every --vis-every iters (and the final iter)
        if args.vis_every > 0 and (it % args.vis_every == 0 or it == args.iters - 1):
            with torch.no_grad():
                save_viz(
                    vis_dir / f"iter_{it:05d}.png",
                    rgb, depth_input, depth_pred, gt_depth, sensor_mask, gt_mask,
                    alpha, beta, loss_val, it,
                )

    print(f"\nDone. total {time.time()-t0:.1f}s")
    print(f"  first loss={first_loss:.5f}, best={best_loss:.5f}, final={loss_val:.5f}")

    # ---- Final sanity verdict ----
    print("\n--- verdict ---")
    if best_loss < 0.1 * first_loss:
        print("PASS: loss dropped >10x from first iter — model can overfit one sample.")
    elif best_loss < 0.5 * first_loss:
        print("OK: loss dropped >2x — overfitting, may need more iters or higher lr.")
    else:
        print("WARN: loss barely dropped. Check for bugs in loss / grads / init.")


if __name__ == "__main__":
    main()
