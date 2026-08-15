#!/usr/bin/env python3
"""
Staged training script for depth completion with resolution progression.

"""
import argparse
import os
import time
import csv
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from encoder import load_backbone, pad_to_multiple, crop_to_original
from fusionv3 import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params
from losses import total_loss
from preprocessed_dataset import PreprocessedHypersimDataset

# Stage configurations
STAGES = [
    {
        "name": "stage1",
        "dataset_stage": 1,
        "iters": 15000,
        "epochs": 10,
    },
    {
        "name": "stage2",
        "dataset_stage": 2,
        "iters": 15000,
        "epochs": 10,
    },
    {
        "name": "stage3",
        "dataset_stage": 3,
        "iters": 10000,
        "epochs": 10,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Staged depth completion training")
    # Data
    p.add_argument("--data-dir", type=str, default="", help="Preprocessed data directory (if set, uses fast preprocessed dataset)")
    p.add_argument("--val-fraction", type=float, default=0.05, help="Validation scene fraction (default: 0.05)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "random", "shared_random"],
                   help="Weight initialization (default: pretrained)")
    # Training
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--enc-lr", type=float, default=1e-5, help="Encoder LR (default: 1e-5)")
    p.add_argument("--dec-lr", type=float, default=1e-4, help="Decoder LR (default: 1e-4)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Grad clip max norm (default: 1.0)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2)")
    p.add_argument("--lr-decay-steps", type=int, default=25000, help="Halve LR every N steps (default: 25000)")
    # Loss
    p.add_argument("--max-samples", type=int, default=10000, help="Max samples per batch for loss (default: 10000)")
    p.add_argument("--near-bias", type=float, default=0.5, help="Near bias for subsampling (default: 0.5)")
    p.add_argument("--local-downsample", type=int, default=4, help="Local loss downsample factor (default: 4)")
    p.add_argument("--level", type=int, default=2, help="Local loss pyramid level (default: 2)")
    p.add_argument("--radius-2d", type=float, default=0.05, help="Local loss 2D radius (default: 0.05)")
    p.add_argument("--min-points", type=int, default=16, help="Min points per patch (default: 16)")
    # Overfitting
    p.add_argument("--overfit", action="store_true", help="Overfit on a single sample (for debugging)")
    p.add_argument("--overfit-sample-idx", type=int, default=0, help="Index of sample to overfit on (default: 0)")
    p.add_argument("--overfit-iters", type=int, default=500, help="Number of iterations for overfitting (default: 500)")
    p.add_argument("--overfit-lr", type=float, default=1e-4, help="Learning rate for overfitting (default: 1e-4)")
    # Checkpointing
    p.add_argument("--out-dir", type=str, default="train_staged_out", help="Output directory (default: train_staged_out)")
    p.add_argument("--save-vis", action="store_true", default=True, help="Save visualizations")
    p.add_argument("--no-save-vis", dest="save_vis", action="store_false", help="Disable visualizations")
    p.add_argument("--save-weights", action="store_true", default=True, help="Save model weights")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", help="Disable weight saving")
    p.add_argument("--ckpt-every", type=int, default=5, help="Checkpoint every N epochs (default: 5)")
    p.add_argument("--vis-every", type=int, default=5, help="Save visualization every N epochs (default: 5)")
    p.add_argument("--val-every-fraction", type=float, default=0.1, help="Run val pass every fraction of stage iters (default: 0.1)")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to search for auto-resume checkpoints (default: checkpoints)")
    # Resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    p.add_argument("--resume-stage", type=int, default=0, help="Stage index to resume from (0, 1, 2)")
    # Misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device (default: cuda if available)")
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers (default: 1)")
    p.add_argument("--dry-run", action="store_true", help="Forward pass only, no training")
    # Overfitting
    p.add_argument("--overfit", action="store_true", help="Overfit on a single sample (for debugging)")
    p.add_argument("--overfit-sample-idx", type=int, default=0, help="Index of sample to overfit on (default: 0)")
    return p.parse_args()


def build_model(args, device):
    """Build encoder and decoder."""
    model_rgb = load_backbone(init_mode=args.init)
    model_depth = load_backbone(init_mode=args.init)
    encoder = DualBranchEncoder(model_rgb, model_depth).to(device)
    decoder = DPTDecoder().to(device)
    return encoder, decoder


def compute_metrics(depth_pred, gt_depth, gt_mask):
    """Compute raw-depth accuracy metrics from pred/gt depth."""
    valid = gt_mask.bool()
    if not valid.any():
        return {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    pred_v = depth_pred[valid]
    gt_v = gt_depth[valid]

    absrel = (pred_v - gt_v).abs() / gt_v.clamp_min(1e-6)
    absrel = absrel.mean().item()

    mae = (pred_v - gt_v).abs().mean().item()
    rmse = torch.sqrt(((pred_v - gt_v) ** 2).mean()).item()

    return {"absrel": absrel, "mae": mae, "rmse": rmse}


def compute_intrinsics(stage_idx: int, height: int, width: int):
    """Scale Hypersim intrinsics to the current stage resolution."""
    scale = {1: 1.0 / 8.0, 2: 1.0 / 5.0, 3: 1.0 / 4.0, 4: 1.0 / 2.0}[stage_idx]
    focal = 886.81
    fx = focal * scale
    fy = focal * scale
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def save_checkpoint(path, *, stage_idx, stage_name, train_step, global_iter, encoder, decoder, optimizer, scheduler, train_loss=None, val_loss=None, val_absrel=None):
    payload = {
        "stage_idx": stage_idx,
        "stage_name": stage_name,
        "train_step": train_step,
        "global_iter": global_iter,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if train_loss is not None:
        payload["train_loss"] = train_loss
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if val_absrel is not None:
        payload["val_absrel"] = val_absrel
    torch.save(payload, path)


def _to_png_name(scene, cam, frame):
    return f"{scene}_{cam}_frame{frame}.png"


def save_depth_map_png(depth_mm: torch.Tensor, path: Path):
    depth = depth_mm.detach().float().cpu().numpy()
    depth = np.squeeze(depth)
    depth = np.clip(depth, 0.0, 65535.0).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth, mode="I;16").save(path)


def _batch_strings(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def run_epoch(
    encoder, decoder, loader, optimizer, scheduler, device, args, alpha, beta,
    stage, phase, max_steps=None, is_train=True, stage_idx=0, stage_dir=None,
    global_iter=0, start_step=0, save_dir=None
):
    """Run one pass (train or eval) over a fixed, non-repeating sample slice."""
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    loss_sum = 0.0
    total_absrel = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_sensor_valid = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            if batch_idx < start_step:
                continue
            if max_steps is not None and batch_idx >= max_steps:
                break

            # Preprocessed data format
            rgb = batch['rgb'].to(device)                    # (B, 3, H, W) in [0,1]
            depth_filled_mm = batch['depth_filled_mm'].to(device)  # (B, 1, H, W) mm
            valid_mask = batch['valid_mask'].to(device)      # (B, H, W) bool
            gt_depth = batch['gt_depth'].to(device)          # (B, 1, H, W) mm
            gt_mask = batch['gt_mask'].to(device)            # (B, 1, H, W) bool

            # Intrinsics from batch (precomputed for padded resolution)
            fx = batch['fx'].to(device) if torch.is_tensor(batch['fx']) else torch.tensor(batch['fx'], device=device)
            fy = batch['fy'].to(device) if torch.is_tensor(batch['fy']) else torch.tensor(batch['fy'], device=device)
            cx = batch['cx'].to(device) if torch.is_tensor(batch['cx']) else torch.tensor(batch['cx'], device=device)
            cy = batch['cy'].to(device) if torch.is_tensor(batch['cy']) else torch.tensor(batch['cy'], device=device)

            # Normalize on-the-fly: zhat = (log(depth_mm) - beta) / alpha
            # depth_filled_mm: (B, 1, H, W) -> squeeze to (B, H, W)
            depth_mm = depth_filled_mm.squeeze(1).clamp_min(1.0)  # (B, H, W)
            zhat = (torch.log(depth_mm) - beta) / alpha  # (B, H, W)
            zhat = zhat.clamp(0.0, 1.0)

            # Build 3-channel input tensor: [zhat, zhat, valid_mask] -> [-1, 1]
            depth_input = torch.stack([zhat, zhat, valid_mask.float()], dim=1) * 2.0 - 1.0  # (B, 3, H, W)

            # Already padded during preprocessing - no pad_to_multiple needed
            orig_h, orig_w = rgb.shape[-2], rgb.shape[-1]
            padded_h, padded_w = orig_h, orig_w
            h_patch, w_patch = padded_h // 16, padded_w // 16

            combined_mask = (gt_mask > 0.5) & valid_mask.unsqueeze(1).bool()
            combined_mask = combined_mask.float()

            features = encoder(rgb, depth_input)
            depth_hat, mask_logit = decoder(features, h_patch, w_patch, padded_h, padded_w)

            # Already at original resolution (no crop_to_original needed)
            depth_pred = denormalize_depth(depth_hat, alpha, beta)

            # Compute loss using combined_mask (GT ∩ sensor)
            loss, parts = total_loss(
                depth_pred, gt_depth, combined_mask, mask_logit, combined_mask,
                fx, fy, cx, cy,
                level=args.level, radius_2d=args.radius_2d,
                min_points_per_patch=args.min_points,
                local_loss_downsample=args.local_downsample,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    max_norm=args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                if stage_dir is not None and args.ckpt_every > 0 and ((batch_idx + 1) % args.ckpt_every == 0 or (max_steps is not None and (batch_idx + 1) == max_steps)):
                    save_checkpoint(
                        stage_dir / "resume.pt",
                        stage_idx=stage_idx,
                        stage_name=stage["name"],
                        train_step=batch_idx + 1,
                        global_iter=global_iter + (batch_idx + 1 - start_step),
                        encoder=encoder,
                        decoder=decoder,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loss=loss.item(),
                    )

            # Metrics on PURE GT mask (not combined)
            metrics = compute_metrics(depth_pred, gt_depth, gt_mask)

            if (not is_train) and save_dir is not None:
                scene_vals = _batch_strings(batch["scene"])
                cam_vals = _batch_strings(batch["cam"])
                frame_vals = _batch_strings(batch["frame"])
                if depth_pred.dim() == 4:
                    pred_maps = depth_pred
                else:
                    pred_maps = depth_pred.unsqueeze(1)
                for i in range(pred_maps.shape[0]):
                    name = _to_png_name(scene_vals[i], cam_vals[i], frame_vals[i])
                    # Save multiple visualizations per-frame: filled input, normalized, sensor-only, and GT
                    stem = name.rsplit('.', 1)[0]
                    try:
                        # Already at original resolution (no crop_to_original needed)
                        depth_input_c = depth_input
                        # pred_maps contains depth_pred per-sample (H,W)
                        # depth_input_c: (B,3,H,W)
                        di = depth_input_c.detach()
                        zhat_all = (di[:, 0, :, :] + 1.0) * 0.5
                        mask_all = ((di[:, 2, :, :] + 1.0) * 0.5) > 0.5
                        for j in range(pred_maps.shape[0]):
                            # filled metric depth (mm)
                            zhat = zhat_all[j]
                            depth_mm_filled = torch.exp(alpha * zhat + beta)
                            save_depth_map_png(depth_mm_filled, save_dir / (stem + f"_filled.png"))

                            # normalized (ZHAT) visualization (8-bit)
                            zhat_vis = (zhat.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                            Image.fromarray(zhat_vis).save(save_dir / (stem + f"_norm.png"))

                            # sensor-only depth (mm)
                            sensor_mask = mask_all[j].float()
                            sensor_depth = (depth_mm_filled * sensor_mask).detach()
                            save_depth_map_png(sensor_depth, save_dir / (stem + f"_sensor.png"))

                            # GT depth (from batch, already mm)
                            gt_map = gt_depth[j].squeeze(0)
                            save_depth_map_png(gt_map, save_dir / (stem + f"_gt.png"))
                    except Exception:
                        # Fall back to saving prediction only
                        save_depth_map_png(pred_maps[i], save_dir / name)

            loss_sum += loss.item()
            total_absrel += metrics['absrel']
            total_mae += metrics['mae']
            total_rmse += metrics['rmse']
            total_sensor_valid += sensor_mask.float().mean().item() * 100
            n_batches += 1

            if is_train and batch_idx % 50 == 0:
                print(f"  [{stage['name']} {phase} B{batch_idx}] loss={loss.item():.4f} "
                      f"absrel={metrics['absrel']:.4f} mae={metrics['mae']:.1f}mm")

    avg_loss = loss_sum / max(1, n_batches)
    avg_absrel = total_absrel / max(1, n_batches)
    avg_mae = total_mae / max(1, n_batches)
    avg_rmse = total_rmse / max(1, n_batches)
    avg_sensor_valid = total_sensor_valid / max(1, n_batches)

    return {
        "loss": avg_loss,
        "absrel": avg_absrel,
        "mae": avg_mae,
        "rmse": avg_rmse,
        "sensor_valid_pct": avg_sensor_valid,
    }


def run_stage(args, encoder, decoder, train_ds, val_ds, device, stage, stage_idx, optimizer, scheduler, global_iter=0, start_step=0):
    """Run one training stage without frame repetition."""
    stage_name = stage["name"]
    stage_budget = stage["iters"]

    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name} | dataset stage: {stage['dataset_stage']} | budget: {stage_budget} iters")
    print(f"{'='*60}")

    # Create data loaders with frame sampling without replacement per epoch
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    alpha, beta = compute_log_params()

    # CSV logging
    stage_dir = Path(args.out_dir) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stage_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "global_iter", "phase", "loss", "absrel", "mae", "rmse",
        "sensor_valid_pct", "lr_enc", "lr_dec"
    ])
    csv_file.flush()

    train_steps = min(stage_budget, len(train_loader))
    start_step = min(start_step, train_steps)
    val_interval = max(1, int(round(train_steps * args.val_every_fraction)))
    print(f"\n--- {stage_name} train: {train_steps} steps over {len(train_ds)} frames (resume at {start_step}) ---")

    consumed = start_step
    latest_train_metrics = None
    latest_val_metrics = None
    while consumed < train_steps:
        next_step = min(train_steps, consumed + val_interval)
        print(f"\n  Train chunk: steps {consumed}..{next_step}")
        latest_train_metrics = run_epoch(
            encoder, decoder, train_loader, optimizer, scheduler, device,
            args, alpha, beta, stage, "train", max_steps=next_step, is_train=True,
            stage_idx=stage_idx, stage_dir=stage_dir, global_iter=global_iter, start_step=consumed
        )
        global_iter += next_step - consumed
        consumed = next_step

        print(f"    Train: loss={latest_train_metrics['loss']:.4f} absrel={latest_train_metrics['absrel']:.4f} "
              f"mae={latest_train_metrics['mae']:.1f}mm rmse={latest_train_metrics['rmse']:.1f}mm")
        csv_writer.writerow([
            consumed, global_iter, "train", latest_train_metrics['loss'], latest_train_metrics['absrel'],
            latest_train_metrics['mae'], latest_train_metrics['rmse'], latest_train_metrics['sensor_valid_pct'],
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
        ])
        csv_file.flush()

        val_run_dir = stage_dir / "val_depth_maps" / f"iter_{global_iter:06d}"
        print(f"  Val pass: saving depth maps to {val_run_dir}")
        latest_val_metrics = run_epoch(
            encoder, decoder, val_loader, optimizer, scheduler, device,
            args, alpha, beta, stage, "val", max_steps=None, is_train=False,
            stage_idx=stage_idx, stage_dir=stage_dir, global_iter=global_iter,
            save_dir=val_run_dir
        )

        print(f"    Val:   loss={latest_val_metrics['loss']:.4f} absrel={latest_val_metrics['absrel']:.4f} "
              f"mae={latest_val_metrics['mae']:.1f}mm rmse={latest_val_metrics['rmse']:.1f}mm")
        csv_writer.writerow([
            consumed, global_iter, "val", latest_val_metrics['loss'], latest_val_metrics['absrel'],
            latest_val_metrics['mae'], latest_val_metrics['rmse'], latest_val_metrics['sensor_valid_pct'],
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
        ])
        csv_file.flush()

        if args.save_weights and latest_val_metrics['absrel'] < float('inf'):
            if not hasattr(run_stage, "_best_val_absrel"):
                run_stage._best_val_absrel = {}
            best_key = (stage_name, stage_idx)
            best_absrel = run_stage._best_val_absrel.get(best_key, float('inf'))
            if latest_val_metrics['absrel'] < best_absrel:
                run_stage._best_val_absrel[best_key] = latest_val_metrics['absrel']
                save_checkpoint(
                    stage_dir / "best_val.pt",
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    train_step=consumed,
                    global_iter=global_iter,
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_absrel=latest_val_metrics['absrel'],
                )
                print(f"    [best val checkpoint saved: absrel={latest_val_metrics['absrel']:.4f}]")

    if args.save_weights:
        save_checkpoint(
            stage_dir / "final.pt",
            stage_idx=stage_idx,
            stage_name=stage_name,
            train_step=train_steps,
            global_iter=global_iter,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=(latest_train_metrics['loss'] if latest_train_metrics is not None else None),
            val_loss=(latest_val_metrics['loss'] if latest_val_metrics is not None else None),
            val_absrel=(latest_val_metrics['absrel'] if latest_val_metrics is not None else None),
        )
        print(f"  [checkpoint saved: final.pt]")

    csv_file.close()
    return global_iter


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("Building dataset index...")
    all_samples = build_hypersim_index(args.scenes_root)
    print(f"Found {len(all_samples)} frames across {len(set(s['scene'] for s in all_samples))} scenes")

    train_samples, val_samples, n_val = split_by_frame(all_samples, args.val_fraction)
    print(f"Train: {len(train_samples)} frames ({len(set(s['scene'] for s in train_samples))} scenes)")
    print(f"Val:   {len(val_samples)} frames ({n_val} frames)")

    stage_samples = split_train_stages(train_samples)

    print("Building model...")
    encoder, decoder = build_model(args, device)

    # Create optimizer & scheduler once (persist across stages)
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.enc_lr},
        {"params": decoder.parameters(), "lr": args.dec_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=0.5)

    global_iter = 0
    start_stage = args.resume_stage
    resume_step = 0

    # Auto-detect latest checkpoint if --resume not provided
    if not args.resume:
        candidates = []
        for d in (Path(args.checkpoints_dir), Path(args.out_dir)):
            try:
                if d.exists():
                    candidates.extend([p for p in d.rglob("*.pt")])
            except Exception:
                pass
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            args.resume = str(latest)
            print(f"Auto-resume: found checkpoint {args.resume}")

    # Handle overfitting mode
    if args.overfit:
        print("\n" + "="*60)
        print("OVERFIT MODE: Training on single sample")
        print("="*60)

        # Use preprocessed dataset for overfitting
        data_dir = args.data_dir if args.data_dir else "/home/khadeeja/vitthing/hypersim_data"
        full_dataset = PreprocessedHypersimDataset(
            data_dir=data_dir,
            stage=args.resume_stage if args.resume_stage > 0 else 1,  # Default to stage 1 if not resuming
            split="train"
        )

        # Get the specific sample to overfit on
        sample_idx = min(args.overfit_sample_idx, len(full_dataset) - 1)
        sample = full_dataset[sample_idx]
        print(f"Overfitting on sample index {sample_idx}: {sample['scene']}/{sample['cam']}/frame.{sample['frame']}")

        # Create single-sample datasets
        overfit_train_ds = torch.utils.data.Subset(full_dataset, [sample_idx])
        overfit_val_ds = torch.utils.data.Subset(full_dataset, [sample_idx])  # Use same sample for validation

        # Temporarily override learning rates for overfitting
        # We need to modify the optimizer's parameter groups directly
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.overfit_lr

        # Run overfitting on a single "stage"
        global_iter = run_stage(
            args, encoder, decoder, overfit_train_ds, overfit_val_ds, device,
            {"name": "overfit", "dataset_stage": args.resume_stage if args.resume_stage > 0 else 1, "iters": args.overfit_iters, "epochs": 1},
            stage_idx=0, optimizer=optimizer, scheduler=scheduler,
            global_iter=0, start_step=0
        )

        # Restore original learning rates
        args.enc_lr = original_enc_lr
        args.dec_lr = original_dec_lr

        print("\n" + "="*60)
        print("OVERFIT COMPLETE!")
        print("="*60)
        return  # Exit after overfitting

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_iter = int(ckpt.get("global_iter", 0))
        start_stage = int(ckpt.get("stage_idx", start_stage))
        resume_step = int(ckpt.get("train_step", 0))
        print(f"Resumed from {args.resume}")

    # Run stages
    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < start_stage:
            continue

        # Use preprocessed dataset
        data_dir = args.data_dir if args.data_dir else "/home/khadeeja/vitthing/hypersim_data"
        train_ds = PreprocessedHypersimDataset(
            data_dir=data_dir,
            stage=stage["dataset_stage"],
            split="train"
        )
        val_ds = PreprocessedHypersimDataset(
            data_dir=data_dir,
            stage=stage["dataset_stage"],
            split="val"
        )

        global_iter = run_stage(
            args, encoder, decoder, train_ds, val_ds, device,
            stage, stage_idx=stage_idx, optimizer=optimizer, scheduler=scheduler,
            global_iter=global_iter, start_step=resume_step if stage_idx == start_stage else 0
        )
        resume_step = 0

    print("\n" + "="*60)
    print("ALL STAGES COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
