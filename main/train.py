from torch.utils.data import Dataset
Dataset.__class_getitem__ = classmethod(lambda cls, *args: cls)
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

from main.encoder import load_backbone, crop_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from main.decoder import DPTDecoder, denormalize_depth
from main.preprocessed_dataset import PreprocessedHypersimDataset
from main.normalize import compute_log_params
from main.intrinsics import get_intrinsics
from main.losses import total_loss

data_dir = str(Path(__file__).resolve().parent.parent / "hypersim_data_simple")
out_dir = str(Path(__file__).resolve().parent.parent / "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=" depth completion training")
    # Data
    p.add_argument("--data-dir", type=str, default=data_dir, help="Preprocessed data directory (if set, uses fast preprocessed dataset)")
    p.add_argument("--val-fraction", type=float, default=0.05, help="Validation scene fraction (default: 0.05)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "random", "shared_random"],
                   help="Weight initialization (default: pretrained)")
    # Training
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--enc_lr", type=float, default=1e-5, help="Encoder LR (default: 1e-5)")
    p.add_argument("--dec_lr", type=float, default=1e-4, help="Decoder LR (default: 1e-4)")
    p.add_argument("--grad-clip", type=float, default=0.1, help="Grad clip max norm (default: 0.1)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2)")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights, train decoder only")
    p.add_argument("--lr-decay-steps", type=int, default=25000, help="Halve LR every N steps (default: 25000)")
    # Loss
    p.add_argument("--no-lg", action="store_true", help="Disable global scale-invariant loss (lg)")
    p.add_argument("--local-downsample", type=int, default=4, help="Local loss downsample factor (default: 4)")
    p.add_argument("--level", type=int, default=2, help="Local loss pyramid level (default: 2)")
    p.add_argument("--radius-2d", type=float, default=0.05, help="Local loss 2D radius (default: 0.05)")
    p.add_argument("--min-points", type=int, default=16, help="Min points per patch (default: 16)")
    # Overfitting
    p.add_argument("--overfit", action="store_true", help="Overfit on a single sample (for debugging)")
    p.add_argument("--overfit-sample-idx", type=int, default=0, help="Index of sample to overfit on (default: 0)")
    p.add_argument("--overfit-iters", type=int, default=500, help="Number of iterations for overfitting (default: 500)")
    p.add_argument("--print-every", type=int, default=50, help="Print overfit metrics every N iterations (default: 50)")
    # Checkpointing
    p.add_argument("--out-dir", type=str, default=out_dir, help="Output directory (default: outputs)")
    p.add_argument("--save-weights", action="store_true", default=True, help="Save model weights")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", help="Disable weight saving")
    p.add_argument("--val-every-fraction", type=float, default=0.1, help="Run val pass every fraction of iters (default: 0.1)")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to search for auto-resume checkpoints (default: checkpoints)")
    # Resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    # Misc
    p.add_argument("--shuffle", action="store_true", help="Shuffle training data")
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers (default: 1)")
    p.add_argument("--dry-run", action="store_true", help="Forward pass only, no training")
    return p.parse_args()


def build_model():
    """Build encoder and decoder."""
    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth).to(DEVICE)
    decoder = DPTDecoder().to(DEVICE)
    return encoder, decoder


def compute_metrics(depth_pred, gt_depth, gt_mask):
    """Compute raw-depth accuracy metrics from pred/gt depth.

    Includes the standard threshold accuracy metrics (delta1/delta2/delta3):
    the fraction of valid pixels where max(pred/gt, gt/pred) falls under
    1.25, 1.25^2, 1.25^3 respectively. delta1 is the usual headline number.

    Also includes delta_1025: the tighter threshold accuracy reported by some
    benchmarks, using the same max(pred/gt, gt/pred) ratio but a 1.025
    threshold instead of 1.25 -- i.e. rel = mean(|pred-gt|/gt) (== absrel
    here) and delta_1025 = fraction of pixels with ratio < 1.025.
    """
    valid = gt_mask.bool()
    if not valid.any():
        return {
            "absrel": float("nan"), "mae": float("nan"), "rmse": float("nan"),
            "delta1": float("nan"), "delta2": float("nan"), "delta3": float("nan"),
            "delta_1025": float("nan"),
        }

    pred_v = depth_pred[valid].clamp_min(1e-6)
    gt_v = gt_depth[valid].clamp_min(1e-6)
    diff = pred_v - gt_v
    abs_diff = diff.abs()

    absrel = (abs_diff / gt_v).mean().item()
    mae = abs_diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()
    delta_1025 = (ratio < 1.025).float().mean().item()

    return {
        "absrel": absrel, "mae": mae, "rmse": rmse,
        "delta1": delta1, "delta2": delta2, "delta3": delta3,
        "delta_1025": delta_1025,
    }


def prepare_sample(sample, device):
    """Preprocess one raw dataset sample into padded, device-resident tensors
    ready for a forward pass. Single-sample (batch dim = 1) only - variable
    resolution across the dataset means samples can't be stacked into a real
    batch, so this is called once per sample and results are accumulated by
    the caller.
    """
    rgb = sample["rgb"].unsqueeze(0)
    gt_depth = sample["gt_depth"].unsqueeze(0) / 1000.0  # mm -> meters
    gt_mask = sample["gt_mask"].unsqueeze(0)
    depth_filled_mm = sample["depth_filled_mm"]
    valid_mask = sample["valid_mask"].float()  # (H, W)

    rgb = rgb.to(device)
    gt_depth = gt_depth.to(device)
    gt_mask = gt_mask.to(device)
    depth_filled_mm = depth_filled_mm.to(device)
    valid_mask = valid_mask.to(device)

    orig_h, orig_w = rgb.shape[-2], rgb.shape[-1]
    fx, fy, cx, cy = get_intrinsics(orig_w, orig_h)

    alpha, beta = compute_log_params(zmin=0.3, zmax=20.0)  # meter range

    depth_m = depth_filled_mm.squeeze(0) / 1000.0  # mm -> meters
    depth_m = depth_m.clamp_min(0.001)  # (H, W)
    zhat = (torch.log(depth_m) - beta) / alpha  # (H, W)
    zhat = zhat.clamp(0.0, 1.0)  # (H, W)

    depth_input_raw = torch.stack([zhat, zhat, valid_mask], dim=0)  # (3, H, W)
    depth_input = depth_input_raw.unsqueeze(0) * 2.0 - 1.0  # (1, 3, H, W)

    rgb_p = crop_to_multiple(rgb)
    depth_input_p = crop_to_multiple(depth_input)
    padded_h, padded_w = rgb_p.shape[-2], rgb_p.shape[-1]
    h_patch, w_patch = padded_h // 16, padded_w // 16  # DINOv3 uses patch size 16

    return {
        "rgb_p": rgb_p,
        "depth_input_p": depth_input_p,
        "gt_depth": gt_depth,
        "gt_mask": gt_mask,
        "valid_mask": valid_mask,
        "alpha": alpha,
        "beta": beta,
        "orig_h": orig_h,
        "orig_w": orig_w,
        "padded_h": padded_h,
        "padded_w": padded_w,
        "h_patch": h_patch,
        "w_patch": w_patch,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
    }


def forward_and_loss(encoder, decoder, prepared, args):
    """Run one forward pass + loss computation on a single prepared sample.
    Returns (loss, parts, depth_pred) so callers can both backprop and log
    accuracy metrics without recomputing the forward pass.
    """
    features = encoder(prepared["rgb_p"], prepared["depth_input_p"])
    depth_hat, mask_logit = decoder(
        features, prepared["h_patch"], prepared["w_patch"],
        prepared["padded_h"], prepared["padded_w"],
    )
    depth_hat = crop_to_original(depth_hat, prepared["orig_h"], prepared["orig_w"])
    mask_logit = crop_to_original(mask_logit, prepared["orig_h"], prepared["orig_w"])
    depth_pred = denormalize_depth(depth_hat, prepared["alpha"], prepared["beta"])

    gt_mask = prepared["gt_mask"]
    valid_mask = prepared["valid_mask"]

    # Combined mask = GT and valid sensor coverage (same as train_staged)

    loss, parts = total_loss(
        depth_pred, prepared["gt_depth"], gt_mask.float(), mask_logit,
        valid_mask.unsqueeze(0).unsqueeze(0),
        prepared["fx"], prepared["fy"], prepared["cx"], prepared["cy"],
        level=args.level, radius_2d=args.radius_2d,
        min_points_per_patch=args.min_points,
        local_loss_downsample=args.local_downsample,
    )

    if args.no_lg and torch.is_tensor(loss):
        loss = loss - torch.tensor(parts["lg"], device=loss.device)
        parts["lg"] = 0.0

    return loss, parts, depth_pred

#allows me to continue training
def save_checkpoint(path, *, train_step, global_iter, encoder, decoder, optimizer, scheduler, train_loss=None, val_loss=None, val_absrel=None, val_delta1=None, val_delta_1025=None):
    payload = {
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
    if val_delta1 is not None:
        payload["val_delta1"] = val_delta1
    if val_delta_1025 is not None:
        payload["val_delta_1025"] = val_delta_1025
    torch.save(payload, path)


def run_validation(encoder, decoder, val_ds, args, save_dir=None, save_n=3):
    """Run a full pass over val_ds and return averaged loss + metrics.

    If save_dir is given, saves depth_pred (and the sigmoid mask prob) as
    .npy for the first `save_n` val samples, for later qualitative analysis.
    """
    encoder.eval()
    decoder.eval()
    total_loss_sum = 0.0
    metric_sums = {"absrel": 0.0, "mae": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "delta_1025": 0.0}
    n = 0
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            prepared = prepare_sample(sample, DEVICE)
            loss, parts, depth_pred = forward_and_loss(encoder, decoder, prepared, args)

            # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
            is_invalid = False
            if torch.is_tensor(loss):
                if torch.isnan(loss) or torch.isinf(loss):
                    is_invalid = True
            else:
                # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
                try:
                    loss_val = float(loss)
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        is_invalid = True
                except (ValueError, TypeError):
                    # If we can't convert to float, treat as invalid
                    is_invalid = True

            if is_invalid:
                print(f"  WARNING: Invalid loss at val sample {i}, skipping (loss type: {type(loss)}, value: {loss})")
                continue

            metrics = compute_metrics(depth_pred, prepared["gt_depth"], prepared["gt_mask"])

            # Safely convert loss to Python float for accumulation
            try:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Could not convert loss to float at val sample {i}, skipping (loss: {loss}, error: {e})")
                continue

            total_loss_sum += loss_value
            for k in metric_sums:
                v = metrics[k]
                if v == v:  # skip NaN (no valid pixels in this sample)
                    metric_sums[k] += v
            n += 1

            if save_dir is not None and i < save_n:
                np.save(save_dir / f"val_sample{i}_depth_mm.npy", depth_pred.detach().cpu().numpy())
                np.save(save_dir / f"val_sample{i}_gt_depth_mm.npy", prepared["gt_depth"].detach().cpu().numpy())
    encoder.train()
    decoder.train()
    if n == 0:
        return float("nan"), {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    avg_loss = total_loss_sum / n
    avg_metrics = {k: v / n for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def find_latest_checkpoint(checkpoints_dir):
    """Return the path to the checkpoint with the highest global_iter in
    checkpoints_dir, or None if the directory doesn't exist or is empty.
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    best_path, best_iter = None, -1
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        it = int(ckpt.get("global_iter", -1))
        if it > best_iter:
            best_iter = it
            best_path = path
    return str(best_path) if best_path is not None else None

def _batch_strings(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def split_by_frame(samples, val_fraction):
    samples = sorted(
        samples,
        key=lambda s: (s["scene"], s["cam"], int(s["frame"]))
    )
    n_val = max(1, int(len(samples) * val_fraction))
    val = samples[-n_val:]
    train = samples[:-n_val]
    return train, val, n_val

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")
    ds = PreprocessedHypersimDataset(data_dir=args.data_dir)
    print(f"Found {len(ds)} frames")

    # Convert dataset to list for train/val split (no shuffle - use original order).
    # Dataset is sorted low-res -> high-res; val is deliberately drawn from the
    # tail so it only contains the higher-resolution samples. Train keeps the
    # full sorted order untouched (no shuffling).
    indices = list(range(len(ds)))

    # Split into train and validation
    val_fraction = args.val_fraction
    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = indices[-n_val:]
    train_indices = indices[:-n_val]

    # Create train and validation datasets
    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")

    print("Building model...")
    encoder, decoder = build_model()

    # clamp cls_token gradients to prevent NaN from attention instability
    def _clip_cls_token(grad):
        return torch.nan_to_num(grad, nan=0.0).clamp(-0.01, 0.01)
    if hasattr(encoder.model_rgb, 'cls_token'):
        encoder.model_rgb.cls_token.register_hook(_clip_cls_token)
    if hasattr(encoder.model_depth, 'cls_token'):
        encoder.model_depth.cls_token.register_hook(_clip_cls_token)

    if args.freeze_encoder:
        print("Freezing encoder weights.")
        for p in encoder.parameters():
            p.requires_grad = False

    # Create optimizer & scheduler once
    trainable_enc = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_enc, "lr": args.enc_lr},
        {"params": decoder.parameters(), "lr": args.dec_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=0.5)

    global_iter = 0
    resume_step = 0

    # Handle overfitting mode
    if args.overfit:
        print("OVERFIT MODE: Training on single sample")
        # Use the full dataset for overfitting (can pick any sample)
        data_dir = args.data_dir

        # Get the specific sample to overfit on
        sample_idx = min(args.overfit_sample_idx, len(ds) - 1)
        sample = ds[sample_idx]
        print(f"Overfitting on sample index {sample_idx}: {sample['scene']}/{sample['cam']}/frame.{sample['frame']}")
        print(f"Sample keys: {list(sample.keys())}")
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"GT depth shape: {sample['gt_depth'].shape}")
        print(f"GT mask shape: {sample['gt_mask'].shape}")
        print(f"Depth filled mm shape: {sample['depth_filled_mm'].shape}")
        print(f"Valid mask shape: {sample['valid_mask'].shape}")

        # Create single-sample datasets
        overfit_train_ds = torch.utils.data.Subset(ds, [sample_idx])
        overfit_val_ds = torch.utils.data.Subset(ds, [sample_idx])  # Use same sample for validation

        optimizer.param_groups[0]['lr'] = args.enc_lr  # Encoder LR
        optimizer.param_groups[1]['lr'] = args.dec_lr  # Decoder LR

        # Run overfitting on a single frame
        print("Starting overfitting...")

        # Extract + preprocess the sample once (padding, intrinsics, normalization)
        prepared = prepare_sample(sample, DEVICE)

        print(f"After padding:")
        print(f"  rgb_padded: {prepared['rgb_p'].shape}")
        print(f"  depth_input_padded: {prepared['depth_input_p'].shape}")
        print(f"  Patches: {prepared['h_patch']}x{prepared['w_patch']}")
        print(f"  GT depth: {prepared['gt_depth'].shape}")
        print(f"  GT mask: {prepared['gt_mask'].shape}")
        print(f"  Valid mask: {prepared['valid_mask'].shape}")
        print(f"  Alpha: {prepared['alpha']:.6f}, Beta: {prepared['beta']:.6f}")
        print(f"  Original size: {prepared['orig_h']}x{prepared['orig_w']}")
        print(f"  Padded size: {prepared['padded_h']}x{prepared['padded_w']}")
        print(f"  FX: {prepared['fx']:.2f}, FY: {prepared['fy']:.2f}, CX: {prepared['cx']:.2f}, CY: {prepared['cy']:.2f}")

        all_params = list(encoder.parameters()) + list(decoder.parameters())

        overfit_pred_dir = Path(args.out_dir) / "predictions" / "overfit"
        overfit_pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(overfit_pred_dir / "gt_depth_mm.npy", prepared["gt_depth"].detach().cpu().numpy())
        save_every = max(1, args.overfit_iters // 10)

        # Training loop
        for it in range(1, args.overfit_iters + 1):
            optimizer.zero_grad()

            loss, parts, depth_pred = forward_and_loss(encoder, decoder, prepared, args)

            # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
            is_invalid = False
            if torch.is_tensor(loss):
                if torch.isnan(loss) or torch.isinf(loss):
                    is_invalid = True
            else:
                # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
                try:
                    loss_val = float(loss)
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        is_invalid = True
                except (ValueError, TypeError):
                    # If we can't convert to float, treat as invalid
                    is_invalid = True

            if is_invalid:
                print(f"  WARNING: Invalid loss at iter {it}, skipping (loss type: {type(loss)}, value: {loss})")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
            optimizer.step()

            # Safely convert loss to Python float
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Could not convert loss to float at iter {it}, skipping (loss: {loss}, error: {e})")
                optimizer.zero_grad()
                continue
            if it == 1:
                first_loss = loss_val

            # Print detailed loss components every print_every iterations
            if it % args.print_every == 0 or it == args.overfit_iters:
                print(f"  Loss components: {parts}")

            if it % save_every == 0 or it == args.overfit_iters:
                np.save(overfit_pred_dir / f"pred_depth_mm_iter{it}.npy", depth_pred.detach().cpu().numpy())

            if it % args.print_every == 0 or it == args.overfit_iters:
                with torch.no_grad():
                    metrics = compute_metrics(depth_pred, prepared["gt_depth"], prepared["gt_mask"])

                print(f"  iter {it:4d}/{args.overfit_iters}  loss={loss_val:.5f}  "
                      f"absrel={metrics['absrel']:.4f}  mae={metrics['mae']:.1f}mm")

        print(f"Saved overfit predictions every {save_every} iters to {overfit_pred_dir}")
        print("OVERFIT COMPLETE!")
        return  # Exit after overfitting

    if args.resume:
        resume_path = args.resume
    else:
        resume_path = find_latest_checkpoint(args.checkpoints_dir)
        if resume_path is not None:
            print(f"Auto-resume: found checkpoint {resume_path}")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_iter = int(ckpt.get("global_iter", 0))
        resume_step = int(ckpt.get("train_step", 0))
        print(f"Resumed from {resume_path} at global_iter={global_iter}, train_step={resume_step}")

    # --- Full training loop (single epoch, no shuffle) ---
    # Samples vary in resolution, so we can't stack them into a real batch
    # tensor. Instead each sample is processed individually (batch dim = 1)
    # and gradients are accumulated over args.batch_size samples before each
    # optimizer.step() - an effective batch size of args.batch_size.
    total_iters = max(1, len(train_ds) // args.batch_size)
    val_every = max(1, int(total_iters * args.val_every_fraction))
    val_every_percent = (val_every / total_iters) * 100
    print(f"Dataset: {len(train_ds)} samples, Batch size: {args.batch_size} → {total_iters} total iterations")
    print(f"Validation every {val_every} iterations ({val_every_percent:.1f}%)")
    print(f"Will validate at iterations: {list(range(val_every, total_iters+1, val_every))[:10]}{'...' if total_iters > val_every*10 else ''}")

    ckpt_dir_path = Path(args.checkpoints_dir)
    if args.save_weights:
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    all_params = list(encoder.parameters()) + list(decoder.parameters())

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=args.shuffle,
        num_workers=args.num_workers, collate_fn=lambda batch: batch[0],
    )

    # If resuming mid-epoch, skip the samples already consumed for this pass.
    samples_to_skip = resume_step * args.batch_size

    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    accum_count = 0
    accum_loss_sum = 0.0
    samples_seen = 0
    consecutive_invalid_losses = 0
    max_consecutive_invalid_losses = 10  # Force step after this many consecutive invalid losses

    for sample in train_loader:
        samples_seen += 1
        if samples_seen <= samples_to_skip:
            continue

        if args.dry_run:
            with torch.no_grad():
                prepared = prepare_sample(sample, DEVICE)
                loss, parts, depth_pred = forward_and_loss(encoder, decoder, prepared, args)
            print(f"[dry-run] sample {samples_seen}/{len(train_ds)}  loss={loss.item():.5f}")
            if samples_seen - samples_to_skip >= args.batch_size:
                break
            continue

        prepared = prepare_sample(sample, DEVICE)
        loss, parts, depth_pred = forward_and_loss(encoder, decoder, prepared, args)

        if global_iter >= 47:
            scene = sample.get('scene', ['?'])[0] if isinstance(sample.get('scene'), list) else sample.get('scene', '?')
            frame = sample.get('frame', ['?'])[0] if isinstance(sample.get('frame'), list) else sample.get('frame', '?')
            print(f"  [diag] iter {global_iter+1} sample: {scene}/frame{frame}")

        # log raw loss every sample for first 20 samples
        if samples_seen <= 20:
            loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            print(f"  [diag] sample {samples_seen} raw loss={loss_val:.4f} parts={parts}")

        # diagnose NaN source
        if torch.is_tensor(loss) and torch.isnan(loss):
            print(f"  [diag] depth_input min={prepared['depth_input_p'].min():.4f} max={prepared['depth_input_p'].max():.4f}")
            print(f"  [diag] pred NaN={torch.isnan(depth_pred).any()} Inf={torch.isinf(depth_pred).any()}")
            for name, p in encoder.named_parameters():
                if torch.isnan(p).any():
                    print(f"  [diag] NaN weight: {name}")
                    break

        # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
        is_invalid = False
        if torch.is_tensor(loss):
            if torch.isnan(loss) or torch.isinf(loss):
                is_invalid = True
        else:
            # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
            try:
                loss_val = float(loss)
                if math.isnan(loss_val) or math.isinf(loss_val):
                    is_invalid = True
            except (ValueError, TypeError):
                # If we can't convert to float, treat as invalid
                is_invalid = True

        if is_invalid:
            # Log detailed information about the problematic sample
            print(f"  WARNING: Invalid loss at iter {global_iter+1}, skipping (loss type: {type(loss)}, value: {loss})")
            print(f"    Sample info: scene={sample['scene']}, cam={sample['cam']}, frame={sample['frame']}")
            print(f"    GT depth range: {prepared['gt_depth'].min():.1f}-{prepared['gt_depth'].max():.1f}mm")
            print(f"    GT mask ratio: {prepared['gt_mask'].float().mean():.4f} ({100*prepared['gt_mask'].float().mean():.2f}%)")
            print(f"    Valid mask ratio: {prepared['valid_mask'].float().mean():.4f} ({100*prepared['valid_mask'].float().mean():.2f}%)")
            # Handle combined mask calculation to avoid CUDA bitwise_and issues
            combined_mask = prepared['gt_mask'].bool() & prepared['valid_mask'].bool()
            print(f"    Combined mask ratio: {combined_mask.float().mean():.4f} ({100*combined_mask.float().mean():.2f}%)")
            print(f"    Pred depth range: {depth_pred.min():.1f}-{depth_pred.max():.1f}mm")
            if torch.is_tensor(depth_pred):
                print(f"    Pred has NaN: {torch.isnan(depth_pred).any()}, Inf: {torch.isinf(depth_pred).any()}")

            consecutive_invalid_losses += 1

            # If we get too many consecutive invalid losses, force a step to prevent getting stuck
            if consecutive_invalid_losses >= max_consecutive_invalid_losses:
                print(f"  WARNING: Too many consecutive invalid losses ({consecutive_invalid_losses}), forcing optimization step")
                # Zero gradients and take a step to prevent getting stuck in accumulation loop
                if accum_count > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_iter += 1
                resume_step = global_iter
                avg_train_loss = accum_loss_sum / max(accum_count, 1)
                print(f"  FORCED STEP: iter {global_iter}  train_loss={avg_train_loss:.5f}")

                accum_loss_sum = 0.0
                accum_count = 0
                consecutive_invalid_losses = 0
            continue

        (loss / args.batch_size).backward()

        # Safely convert loss to Python float for accumulation
        try:
            loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
        except (ValueError, TypeError) as e:
            print(f"  WARNING: Could not convert loss to float at iter {global_iter+1}, skipping (loss: {loss}, error: {e})")
            optimizer.zero_grad()
            continue

        accum_loss_sum += loss_value
        accum_count += 1

        if accum_count == args.batch_size:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
            if global_iter % 10 == 0:
                print(f"  [diag] grad_norm={grad_norm:.4f}")
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # check for NaN weights after step
            for name, p in encoder.named_parameters():
                if torch.isnan(p).any():
                    raise RuntimeError(f"NaN weights detected in {name} at iter {global_iter+1}")

            global_iter += 1
            resume_step = global_iter
            avg_train_loss = accum_loss_sum / accum_count
            accum_loss_sum = 0.0
            accum_count = 0

            if global_iter % max(1, val_every // 5) == 0 or global_iter == total_iters:
                # Calculate next validation iteration
                next_val = ((global_iter // val_every) + 1) * val_every
                if next_val > total_iters:
                    next_val_str = "end"
                else:
                    next_val_str = f"{next_val}"
                print(f"iter {global_iter:5d}/{total_iters}  train_loss={avg_train_loss:.5f}  [next val at iter {next_val_str}]")

            if global_iter % val_every == 0 or global_iter == total_iters:
                print(f"--- Running validation at iter {global_iter} ---")
                pred_save_dir = Path(args.out_dir) / "predictions" / f"iter{global_iter}"
                val_loss, val_metrics = run_validation(encoder, decoder, val_ds, args, save_dir=pred_save_dir)
                print(f"  [val @ iter {global_iter}] val_loss={val_loss:.5f}  "
                      f"absrel={val_metrics['absrel']:.4f}  mae={val_metrics['mae']:.1f}mm  "
                      f"rmse={val_metrics['rmse']:.1f}mm  "
                      f"d1={val_metrics['delta1']:.4f}  d2={val_metrics['delta2']:.4f}  d3={val_metrics['delta3']:.4f}  "
                      f"d1.025={val_metrics['delta_1025']:.4f}")
                print(f"  [val predictions saved: {pred_save_dir}]")

                if args.save_weights:
                    ckpt_path = ckpt_dir_path / f"ckpt_iter{global_iter}.pt"
                    save_checkpoint(
                        ckpt_path, train_step=global_iter, global_iter=global_iter,
                        encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                        train_loss=avg_train_loss, val_loss=val_loss, val_absrel=val_metrics["absrel"],
                        val_delta1=val_metrics["delta1"], val_delta_1025=val_metrics["delta_1025"],
                    )
                    print(f"  [checkpoint saved: {ckpt_path}]")

            if global_iter >= total_iters:
                break

    print("TRAINING COMPLETE!")
    if args.save_weights:
        final_ckpt_path = ckpt_dir_path / "ckpt_final.pt"
        save_checkpoint(
            final_ckpt_path, train_step=global_iter, global_iter=global_iter,
            encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
        )
        print(f"[final checkpoint saved: {final_ckpt_path}]")


if __name__ == "__main__":
    main()
