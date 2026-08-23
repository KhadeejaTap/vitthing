#!/usr/bin/env python3
"""
Overfit on a specific NPZ sample: hypersim_data/stage1/train/ai_001_001_scene_cam_00_frame0000.npz

Sanity check: if the model can't drive the loss to ~0 on one sample,
something is broken (loss bug, detached gradients, bad init, etc.).
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

from main.encoder import load_backbone, pad_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params
from losses import total_loss


def parse_args():
    p = argparse.ArgumentParser(description="Overfit one specific NPZ sample for N iters")
    p.add_argument("--npz-path", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "hypersim_data" / "stage1" / "train" / "ai_001_001_scene_cam_00_frame0000.npz"),
                   help="Path to the specific NPZ file to overfit on")
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
    p.add_argument("--vis-dir", type=str, default="overfit_specific_vis",
                   help="Directory for visualization PNGs (default: overfit_specific_vis)")
    return p.parse_args()


def _depth_to_color(d, vmin, vmax):
    """Normalize depth to [0,1] and apply colormap. d: (H,W) tensor -> (H,W,3) numpy."""
    _TURBO = LinearSegmentedColormap.from_list(
        "turbo_approx",
        ["#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4",
         "#24eca6", "#61fc6c", "#a4fc3b", "#d1e834", "#f6c438",
         "#fb8b32", "#f65322", "#cb1b1b", "#7d0a4e", "#30123b"],
    )
    d = d.detach().cpu().float()
    d = (d - vmin) / max(vmax - vmin, 1e-6)
    d = d.clamp(0, 1).numpy()
    return (_TURBO(d)[..., :3] * 255).astype(np.uint8)


def save_viz(path, rgb, depth_input, depth_pred, gt_depth, sensor_mask, gt_mask, mask_logit,
             alpha, beta, loss_val, it):
    """
    Save a 6-panel visualization: RGB | GT depth | Pred depth | Sensor input | Pred Mask | |Pred-GT| error.
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

    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    panels = [
        ("RGB", rgb_img.clip(0, 1), None),
        ("GT depth (mm)", _depth_to_color(gt, vmin, vmax), None),
        ("Pred depth (mm)", _depth_to_color(pred, vmin, vmax), None),
        ("Sensor input (yellow=real)", sensor_color, None),
        ("Pred Mask (prob)", torch.sigmoid(mask_logit)[0, 0].detach().cpu().numpy(), None),
        ("|Pred - GT| (mm)", err_img.numpy(), "magma"),
    ]
    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap is None and img.ndim == 3:
            ax.imshow(img)
        elif cmap is None:
            # Special handling for mask panel: use fixed [0,1] range
            if title == "Pred Mask (prob)":
                ax.imshow(img, vmin=0, vmax=1)
            else:
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
    import numpy as np
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading specific NPZ file: {args.npz_path}")

    # Load the specific NPZ file directly
    data = np.load(args.npz_path, allow_pickle=True)

    # Extract data from NPZ (matching PreprocessedHypersimDataset format)
    rgb = data["rgb"].astype(np.float32)           # (H, W, 3) [0,1]
    gt_depth = data["gt_depth"].astype(np.float32)  # (H, W) mm
    gt_mask = data["gt_mask"].astype(np.bool_)      # (H, W) bool
    sensor_depth = data["sensor_depth"].astype(np.float32)  # (H, W) mm padded
    sensor_mask = data["sensor_mask"].astype(np.bool_)      # (H, W) bool padded
    crop_perimeter = data["crop_perimeter"]         # [y, x, h, w]
    sensor_option = data["sensor_option"]
    if isinstance(sensor_option, np.ndarray):
        sensor_option = sensor_option.item()
    meta = data["meta"]
    if isinstance(meta, np.ndarray):
        meta = meta.tolist()

    print(f"Loaded sample: scene={meta[0]}, cam={meta[1]}, frame={meta[2]}, stage={meta[3]}")
    print(f"RGB shape: {rgb.shape}")
    print(f"GT depth shape: {gt_depth.shape}")
    print(f"Sensor depth shape: {sensor_depth.shape}")
    print(f"Sensor mask shape: {sensor_mask.shape}")

    # Convert to tensors and move to device
    # rgb: (H, W, 3) -> (1, 3, H, W)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)

    # gt_depth: (H, W) -> (1, 1, H, W)
    gt_depth_tensor = torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).contiguous().to(device)

    # gt_mask: (H, W) -> (1, 1, H, W)
    gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).contiguous().to(device)

    # sensor_depth: (H, W) -> (1, 1, H, W) - this is the flood-filled input
    sensor_depth_tensor = torch.from_numpy(sensor_depth).unsqueeze(0).unsqueeze(0).contiguous().to(device)

    # sensor_mask: (H, W) -> (1, 1, H, W) - sparse validity mask
    sensor_mask_tensor = torch.from_numpy(sensor_mask).unsqueeze(0).unsqueeze(0).contiguous().to(device)

    print(f"Tensor shapes after conversion:")
    print(f"  rgb_tensor: {rgb_tensor.shape}")
    print(f"  gt_depth_tensor: {gt_depth_tensor.shape}")
    print(f"  gt_mask_tensor: {gt_mask_tensor.shape}")
    print(f"  sensor_depth_tensor: {sensor_depth_tensor.shape}")
    print(f"  sensor_mask_tensor: {sensor_mask_tensor.shape}")

    # For depth_input to encoder, we need to build the 3-channel input:
    # [zhat, zhat, valid_mask] -> [-1, 1] where zhat = (log(depth_mm) - beta) / alpha
    alpha, beta = compute_log_params()
    depth_mm = sensor_depth_tensor.squeeze(1).squeeze(1).clamp_min(1.0)  # (1, H, W)
    zhat = (torch.log(depth_mm) - beta) / alpha  # (1, H, W)
    zhat = zhat.clamp(0.0, 1.0)  # (1, H, W)

    print(f"Debug shapes: depth_mm={depth_mm.shape}, zhat={zhat.shape}, sensor_mask_tensor={sensor_mask_tensor.shape}")

    # Build 3-channel input tensor: [zhat, zhat, valid_mask] -> [-1, 1]
    # Ensure all tensors have shape (1, H, W) before stacking along dim=1
    zhat_ch1 = zhat  # (1, H, W)
    zhat_ch2 = zhat  # (1, H, W) - second copy for ch1
    sensor_mask_ch = sensor_mask_tensor.float()  # (1, H, W)

    print(f"Debug channel shapes: zhat_ch1={zhat_ch1.shape}, zhat_ch2={zhat_ch2.shape}, sensor_mask_ch={sensor_mask_ch.shape}")

    depth_input = torch.stack([zhat_ch1, zhat_ch2, sensor_mask_ch], dim=1)  # (1, 3, H, W)
    depth_input = depth_input * 2.0 - 1.0  # (1, 3, H, W)
    print(f"Final depth_input shape: {depth_input.shape}")

    print(f"Prepared tensors:")
    print(f"  rgb: {rgb_tensor.shape}")
    print(f"  depth_input: {depth_input.shape}")
    print(f"  gt_depth: {gt_depth_tensor.shape}")
    print(f"  gt_mask: {gt_mask_tensor.shape}")
    print(f"  sensor_mask: {sensor_mask_tensor.shape}")

    # combined mask = GT ∩ sensor (same as train_staged)
    combined_mask = (gt_mask_tensor > 0.5) & sensor_mask_tensor.unsqueeze(1).bool()
    combined_mask = combined_mask.float()

    orig_h, orig_w = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
    print(f"Original resolution: {orig_h}x{orig_w}")

    # Pad to patch multiple (frozen — same padded size every iter)
    rgb_p = pad_to_multiple(rgb_tensor)
    depth_input_p = pad_to_multiple(depth_input)
    padded_h, padded_w = rgb_p.shape[-2], rgb_p.shape[-1]
    h_patch, w_patch = padded_h // 16, padded_w // 16  # DINOv3 uses patch size 16

    print(f"After padding:")
    print(f"  rgb_padded: {rgb_p.shape}")
    print(f"  depth_input_padded: {depth_input_p.shape}")
    print(f"  Patches: {h_patch}x{w_patch}")

    # ---- Sensor data & ground truth: shapes + ranges ----
    sensor_valid = sensor_mask_tensor > 0.5
    gt_valid = gt_mask_tensor > 0.5

    print("\n--- sensor data ---")
    print(f"  depth_input: shape={tuple(depth_input.shape)}, "
          f"range=[{depth_input.min().item():.4f}, {depth_input.max().item():.4f}] "
          f"(expect ~[-1,1])")
    # depth_input ch2 is the validity mask in [-1,1] encoding; decode it
    di_mask = (depth_input[:, 2] + 1) / 2  # [-1,1] -> [0,1]
    print(f"  depth_input ch2 (mask decoded): "
          f"{int((di_mask > 0.5).sum().item())} valid px "
          f"({100*(di_mask > 0.5).float().mean().item():.2f}%)")
    print(f"  sensor_mask: shape={tuple(sensor_mask_tensor.shape)}, "
          f"{int(sensor_valid.sum().item())} valid px "
          f"({100*sensor_valid.float().mean().item():.2f}%)")

    print("\n--- ground truth ---")
    print(f"  gt_depth: shape={tuple(gt_depth_tensor.shape)}, dtype={gt_depth_tensor.dtype}, "
          f"range=[{gt_depth_tensor[gt_valid].min().item():.1f}, "
          f"{gt_depth_tensor[gt_valid].max().item():.1f}] mm "
          f"(all px: [{gt_depth_tensor.min().item():.1f}, {gt_depth_tensor.max().item():.1f}])")
    print(f"  gt_mask: shape={tuple(gt_mask_tensor.shape)}, "
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
            depth_pred, gt_depth_tensor, combined_mask, mask_logit, sensor_mask_tensor,
            # For fx, fy, cx, cy we need to compute from the original resolution
            886.81 * (1.0/8.0), 886.81 * (1.0/8.0), (orig_w-1)/2.0, (orig_h-1)/2.0,  # stage1 intrinsics
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
                valid = gt_mask_tensor.bool()
                if valid.any():
                    pv = depth_pred[valid]
                    gv = gt_depth_tensor[valid]
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
                    rgb_tensor, depth_input, depth_pred, gt_depth_tensor, sensor_mask_tensor, gt_mask_tensor, mask_logit,
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