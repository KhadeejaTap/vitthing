#!/usr/bin/env python3
"""
Dynamic test script for depth completion overfitting using Hypersim dataset.
Usage: python test.py [options]
"""
import argparse
import os
import time
import csv
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from encoder import load_backbone, pad_to_multiple, crop_to_original
from fusion import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params
from losses import total_loss
from main.dataset import HypersimDepthCompletionDataset, TARGET_RESOLUTIONS, RESOLUTION_WEIGHTS


def parse_args():
    p = argparse.ArgumentParser(description="Depth completion overfit test")
    # Data
    p.add_argument("--scene-dir", type=str, default="/home/khadeeja/ml-hypersim/evermotion_dataset/scenes/ai_001_001",
                   help="Hypersim scene directory")
    p.add_argument("--frame-indices", type=str, default="0", help="Comma-separated frame indices (default: 0)")
    p.add_argument("--camera", type=str, default="cam_00", help="Camera name (default: cam_00)")
    # Resolution
    p.add_argument("--res", type=str, default="224x224", help="Resolution HxW (default: 224x224)")
    p.add_argument("--multi-res", action="store_true", help="Sample random resolution each iteration")
    # Model
    p.add_argument("--backbone", type=str, default="vit_small_reg4", help="DINOv2 backbone")
    p.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "random", "shared_random"],
                   help="Weight initialization (default: pretrained)")
    p.add_argument("--decoder-channels", type=str, default="384,256,64,32,16",
                   help="Decoder channel schedule comma-separated")
    # Training
    p.add_argument("--iters", type=int, default=500, help="Training iterations (default: 500)")
    p.add_argument("--enc-lr", type=float, default=1e-5, help="Encoder LR (default: 1e-5, paper)")
    p.add_argument("--dec-lr", type=float, default=1e-4, help="Decoder LR (default: 1e-4, paper)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Grad clip max norm (default: 1.0, paper)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2, paper)")
    p.add_argument("--lr-decay-steps", type=int, default=25000, help="Halve LR every N steps (default: 25000, paper)")
    # Sensor simulation
    p.add_argument("--sensor-h", type=int, default=480, help="Sensor height (default: 480)")
    p.add_argument("--sensor-w", type=int, default=640, help="Sensor width (default: 640)")
    p.add_argument("--sensor-fx", type=float, default=544.462653, help="Sensor fx (default: 544.46)")
    p.add_argument("--rgb-h", type=int, default=1080, help="RGB height (default: 1080)")
    p.add_argument("--rgb-w", type=int, default=1920, help="RGB width (default: 1920)")
    p.add_argument("--rgb-fx", type=float, default=910.799450, help="RGB fx (default: 910.80)")
    p.add_argument("--sparsity-level", type=str, default="random", choices=["low", "medium", "high", "random"],
                   help="Sparsity level: low(15%), medium(30%), high(50%), random (default: random)")
    p.add_argument("--sensor-jitter", type=float, default=0.15, help="Sensor position jitter (default: 0.15)")
    p.add_argument("--edge-falloff", type=float, default=0.2, help="Edge falloff sparsity (default: 0.2)")
    p.add_argument("--falloff-power", type=float, default=2.0, help="Falloff power (default: 2.0)")
    p.add_argument("--depth-sparsity-scale", type=float, default=0.3, help="Depth-dependent sparsity scale (default: 0.3)")
    # Loss
    p.add_argument("--max-samples", type=int, default=10000, help="Max samples per batch for loss (default: 10000)")
    p.add_argument("--near-bias", type=float, default=0.5, help="Near bias for subsampling 0-1 (default: 0.5)")
    p.add_argument("--local-downsample", type=int, default=4, help="Local loss downsample factor (default: 4)")
    p.add_argument("--level", type=int, default=2, help="Local loss pyramid level (default: 2)")
    p.add_argument("--radius-2d", type=float, default=0.05, help="Local loss 2D radius (default: 0.05)")
    p.add_argument("--min-points", type=int, default=16, help="Min points per patch (default: 16)")
    # Checkpointing
    p.add_argument("--ckpt-every", type=int, default=50, help="Checkpoint every N iters (default: 50)")
    p.add_argument("--vis-every", type=int, default=500, help="Save visualization every N iters (default: 500)")
    p.add_argument("--out-dir", type=str, default="test_out", help="Output directory (default: test_out)")
    p.add_argument("--save-vis", action="store_true", default=True, help="Save visualizations")
    p.add_argument("--no-save-vis", dest="save_vis", action="store_false", help="Disable visualizations")
    p.add_argument("--save-weights", action="store_true", default=True, help="Save model weights")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", help="Disable weight saving")
    # Resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    # Misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device (default: cuda if available)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--dry-run", action="store_true", help="Forward pass only, no training")
    return p.parse_args()


def build_model(args, device):
    """Build encoder and decoder."""
    model_rgb = load_backbone(init_mode=args.init)
    model_depth = load_backbone(init_mode=args.init)
    encoder = DualBranchEncoder(model_rgb, model_depth).to(device)

    channels = list(map(int, args.decoder_channels.split(",")))
    decoder = DPTDecoder().to(device)
    if channels != decoder.CHANNELS:
        print(f"Warning: decoder channels {channels} != default {decoder.CHANNELS}, using default")

    return encoder, decoder


def compute_metrics(depth_pred, gt_depth, gt_mask):
    """Compute raw-depth accuracy metrics from pred/gt depth."""
    valid = gt_mask.bool()
    if not valid.any():
        return {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    pred_v = depth_pred[valid]
    gt_v = gt_depth[valid]

    # absrel: mean absolute relative error
    absrel = (pred_v - gt_v).abs() / gt_v.clamp_min(1e-6)
    absrel = absrel.mean().item()

    # mae: mean absolute error (mm)
    mae = (pred_v - gt_v).abs().mean().item()

    # rmse: root mean squared error (mm)
    rmse = torch.sqrt(((pred_v - gt_v) ** 2).mean()).item()

    return {"absrel": absrel, "mae": mae, "rmse": rmse}


def run_training(args, encoder, decoder, dataset, device):
    """Run training loop."""
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
    if args.save_weights:
        os.makedirs(os.path.join(args.out_dir, "weights"), exist_ok=True)

    # CSV logging
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "iter", "loss_total", "l1", "lg", "ll", "lm",
        "absrel", "mae", "rmse", "lg_delta", "ll_delta", "time_ms",
        "res_h", "res_w", "sensor_valid_pct"
    ])
    csv_file.flush()

    alpha, beta = compute_log_params()

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.enc_lr},
        {"params": decoder.parameters(), "lr": args.dec_lr},
    ], weight_decay=args.weight_decay)

    # LR scheduler: halve every lr_decay_steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=0.5)

    # Resume from checkpoint
    start_iter = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_iter = ckpt["iteration"] + 1
        print(f"Resumed from {args.resume} at iteration {start_iter-1}")

    print(f"Training: {args.iters} iters")
    print(f"Dataset: {len(dataset)} frames from {args.scene_dir}")
    print(f"Resolution: {'multi-res' if args.multi_res else args.res}")
    print(f"Sensor: {args.sensor_w}x{args.sensor_h} (fx={args.sensor_fx}) -> {args.rgb_w}x{args.rgb_h} (fx={args.rgb_fx})")
    print(f"  sparsity={args.sparsity_level}, jitter={args.sensor_jitter}, edge_falloff={args.edge_falloff}, depth_sparsity={args.depth_sparsity_scale}")
    print(f"Loss params: max_samples={args.max_samples}, near_bias={args.near_bias}, "
          f"local_downsample={args.local_downsample}, level={args.level}")
    print(f"Logging to: {csv_path}")
    print()

    for it in range(start_iter, args.iters + 1):
        t0 = time.time()
        optimizer.zero_grad()

        # Get sample (dataset handles resolution sampling if multi_res)
        sample = dataset[0]  # Single frame overfit test
        rgb = sample['rgb'].unsqueeze(0).to(device)
        depth_input = sample['depth_input'].unsqueeze(0).to(device)
        gt_depth = sample['gt_depth'].unsqueeze(0).to(device)
        gt_mask = sample['gt_mask'].unsqueeze(0).to(device)
        sensor_mask = sample['sensor_mask'].unsqueeze(0).to(device)
        fx, fy, cx, cy = sample['intrinsics']
        meta = sample['meta']

        out_h, out_w = rgb.shape[-2], rgb.shape[-1]
        h_patch, w_patch = out_h // 14, out_w // 14

        features = encoder(rgb, depth_input)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        # Compute loss at padded resolution (no crop needed since gt is already padded)
        depth_pred = denormalize_depth(depth_hat, alpha, beta)

        loss, parts = total_loss(
            depth_pred, gt_depth, gt_mask, mask_logit, gt_mask,
            fx, fy, cx, cy,
            level=args.level, radius_2d=args.radius_2d,
            min_points_per_patch=args.min_points,
        )

        if args.dry_run:
            print(f"iter {it}: total={loss.item():.4f}  l1={parts['l1']:.4f}  "
                  f"lg={parts['lg']:.4f}  ll={parts['ll']:.4f}  lm={parts['lm']:.4f}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
        optimizer.step()
        scheduler.step()

        t1 = time.time()

        # Compute raw-depth metrics
        metrics = compute_metrics(depth_pred, gt_depth, gt_mask)

        # Extract lg_delta, ll_delta from parts if available
        lg_delta = parts.get("lg_delta", 0.0)
        ll_delta = parts.get("ll_delta", 0.0)

        sensor_valid_pct = sensor_mask.float().mean().item() * 100

        print(f"iter {it:4d}: total={loss.item():.4f}  l1={parts['l1']:.4f}  "
              f"lg={parts['lg']:.4f}  ll={parts['ll']:.4f}  lm={parts['lm']:.4f}  "
              f"absrel={metrics['absrel']:.4f}  mae={metrics['mae']:.2f}mm  "
              f"rmse={metrics['rmse']:.2f}mm  res={out_h}x{out_w}  "
              f"sensor_valid={sensor_valid_pct:.1f}%  time={(t1-t0)*1000:.1f}ms")

        # Log to CSV
        csv_writer.writerow([
            it, loss.item(), parts['l1'], parts['lg'], parts['ll'], parts['lm'],
            metrics['absrel'], metrics['mae'], metrics['rmse'],
            lg_delta, ll_delta, (t1-t0)*1000,
            out_h, out_w, sensor_valid_pct
        ])
        csv_file.flush()

        # Checkpoint
        if it == 1 or it % args.ckpt_every == 0 or it == args.iters:
            if args.save_weights:
                with torch.no_grad():
                    torch.save({
                        "iteration": it,
                        "encoder_state_dict": encoder.state_dict(),
                        "decoder_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                    }, os.path.join(args.out_dir, "weights", f"overfit_iter{it}.pt"))
                print(f"  [weights saved at iter {it}]")

        # Save depth/mask npy every vis_every (inference artifacts)
        if it == 1 or it % args.vis_every == 0 or it == args.iters:
            with torch.no_grad():
                depth_np = depth_pred.detach().cpu().numpy().copy()
                mask_np = torch.sigmoid(mask_logit).detach().cpu().numpy().copy()
                os.makedirs(os.path.join(args.out_dir, "depth_maps"), exist_ok=True)
                np.save(os.path.join(args.out_dir, "depth_maps", f"pred_depth_iter{it}.npy"), depth_np)
                np.save(os.path.join(args.out_dir, "depth_maps", f"pred_mask_iter{it}.npy"), mask_np)
            print(f"  [depth/mask saved at iter {it}]")

        # Save visualizations (PNG) only if save_vis
        if args.save_vis and (it == 1 or it % args.vis_every == 0 or it == args.iters):
            with torch.no_grad():
                depth_np = depth_pred.detach().cpu().numpy().copy()
                mask_np = torch.sigmoid(mask_logit).detach().cpu().numpy().copy()
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                valid_d = depth_np > 0
                vmin = depth_np[valid_d].min() if valid_d.any() else 0
                vmax = depth_np[valid_d].max() if valid_d.any() else 1
                norm = np.clip((depth_np.squeeze() - vmin) / (vmax - vmin + 1e-6), 0, 1)
                colored = cm.turbo(norm)[..., :3]
                plt.imsave(os.path.join(args.out_dir, "vis", f"pred_depth_iter{it}.png"), colored)
                plt.imsave(os.path.join(args.out_dir, "vis", f"pred_mask_iter{it}.png"),
                           mask_np.squeeze(), cmap="gray")
            print(f"  [visualization saved at iter {it}]")

        if torch.isnan(loss):
            print(f"  NaN at iter {it}! Stopping.")
            break

    csv_file.close()
    print(f"Done! Metrics logged to {csv_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Parse frame indices
    frame_indices = [int(x.strip()) for x in args.frame_indices.split(",")]

    # Parse resolution
    target_h, target_w = map(int, args.res.split("x"))

    print("Loading dataset...")
    dataset = HypersimDepthCompletionDataset(
        scene_dir=Path(args.scene_dir),
        frame_indices=frame_indices,
        target_resolution=(target_h, target_w),
        camera=args.camera,
        multi_res=args.multi_res,
        sensor_h=args.sensor_h,
        sensor_w=args.sensor_w,
        sensor_fx=args.sensor_fx,
        rgb_h=args.rgb_h,
        rgb_w=args.rgb_w,
        rgb_fx=args.rgb_fx,
        sparsity_level=args.sparsity_level,
        sensor_pos_jitter=args.sensor_jitter,
        edge_falloff=args.edge_falloff,
        falloff_power=args.falloff_power,
        depth_sparsity_scale=args.depth_sparsity_scale,
        augment=True,
    )
    print(f"Loaded {len(dataset)} frames")

    print("Building model...")
    encoder, decoder = build_model(args, device)

    run_training(args, encoder, decoder, dataset, device)
    print("Done!")


if __name__ == "__main__":
    main()
