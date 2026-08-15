#!/usr/bin/env python3
"""
Low-resolution test: compare overfit at 224x224 (DINOv2 native) vs 546x966 (current).
Downsamples all frame_0000 data consistently before padding.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from encoder import load_backbone, pad_to_multiple, crop_to_original
from fusion import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params, build_input_tensor
from intrinsics import get_intrinsics, get_intrinsics_for_res
from losses import total_loss, depth_weighted_l1_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ITERS = 500
ENCODER_LR = 1e-4
DECODER_LR = 5e-4  # lowered from 1e-3 for wider decoder stability
GRAD_CLIP_MAX_NORM = 0.5  # tighter clipping
CHECKPOINT_EVERY = 50  # save every N iterations

# Target resolution
LOW_RES = (224, 224)      # DINOv2 native


def center_crop(arr, target_h, target_w):
    """Crop center region of array."""
    h, w = arr.shape[:2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    if arr.ndim == 3:
        return arr[top:top+target_h, left:left+target_w, :]
    else:
        return arr[top:top+target_h, left:left+target_w]


def load_frame_0000():
    """Load all frame_0008 data at original 540x960."""
    rgb = np.array(Image.open('data/frame_0008_rgb.png')).astype(np.float32) / 255.0
    depth = np.load('data/frame_0008_depth_proj_mm.npy').astype(np.float32)
    mask = np.load('data/frame_0008_proj_valid_mask.npy').astype(np.float32)
    gt_depth = np.load('data/frame_0008_gt_mm.npy').astype(np.float32)
    gt_mask = np.load('data/frame_0008_gt_mask.npy').astype(np.float32)
    return rgb, depth, mask, gt_depth, gt_mask


def build_sample(rgb, depth, mask, gt_depth, gt_mask, alpha, beta):
    """Build sample dict matching DToFDataset format."""
    depth_tensor = build_input_tensor(depth, mask, alpha, beta)
    return {
        "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
        "depth_input": torch.from_numpy(depth_tensor),
        "valid_mask": torch.from_numpy(mask),
        "gt_depth": torch.from_numpy(gt_depth).unsqueeze(0),
        "gt_mask": torch.from_numpy(gt_mask).unsqueeze(0),
        "alpha": alpha,
        "beta": beta,
    }


def run_overfit(resolution_name, sample, orig_h, orig_w, out_dir):
    """Run overfit test for a given resolution."""
    print(f"\n{'='*60}")
    print(f"  OVERFIT TEST: {resolution_name} ({orig_h}x{orig_w})")
    print(f"{'='*60}")

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0)).to(DEVICE)
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0)).to(DEVICE)
    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 14, out_w // 14
    print(f"Padded: {out_h}x{out_w}, Patch grid: {h_patch}x{w_patch}")

    gt_depth = sample["gt_depth"].unsqueeze(0).to(DEVICE)
    gt_mask = sample["gt_mask"].unsqueeze(0).to(DEVICE)

    # Compute intrinsics for this resolution (before padding), then adjust for padding
    fx_s, fy_s, cx_s, cy_s = get_intrinsics_for_res(orig_h, orig_w)
    # pad_to_multiple adds symmetric padding, so adjust principal point
    pad_h = out_h - orig_h
    pad_w = out_w - orig_w
    cx_s += pad_w // 2
    cy_s += pad_h // 2
    print(f"Intrinsics: fx={fx_s:.2f} fy={fy_s:.2f} cx={cx_s:.2f} cy={cy_s:.2f}")

    alpha, beta = sample["alpha"], sample["beta"]

    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth).to(DEVICE)
    decoder = DPTDecoder().to(DEVICE)

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": decoder.parameters(), "lr": DECODER_LR},
    ])

    checkpoints = {}
    weights_dir = os.path.join(out_dir, "overfit_weights")
    os.makedirs(weights_dir, exist_ok=True)

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        features = encoder(rgb, depth_input)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        depth_hat_c = crop_to_original(depth_hat, orig_h, orig_w)
        mask_logit_c = crop_to_original(mask_logit, orig_h, orig_w)

        depth_pred_metric = denormalize_depth(depth_hat_c, alpha, beta)

        loss, parts = total_loss(
            depth_pred_metric, gt_depth, gt_mask, mask_logit_c, gt_mask,
            fx_s, fy_s, cx_s, cy_s
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if it == 1 or it % CHECKPOINT_EVERY == 0 or it == NUM_ITERS:
            with torch.no_grad():
                valid = gt_mask.bool()
                if valid.any():
                    abs_err = (depth_pred_metric - gt_depth).abs()[valid].mean().item()
                else:
                    abs_err = float("nan")
            print(f"iter {it:4d}  total={loss.item():.4f}  l1={parts['l1']:.4f}  mean_abs_err={abs_err:.2f}mm")

        # Debug NaN
        if torch.isnan(loss):
            print(f"  NaN at iter {it}! depth_pred range: {depth_pred_metric.min().item():.2f} - {depth_pred_metric.max().item():.2f}")
            print(f"  gt_depth range: {gt_depth.min().item():.2f} - {gt_depth.max().item():.2f}")
            print(f"  depth_hat range: {depth_hat.min().item():.2f} - {depth_hat.max().item():.2f}")
            print(f"  alpha={alpha}, beta={beta}")
            break

        if it == 1 or it % CHECKPOINT_EVERY == 0 or it == NUM_ITERS:
            with torch.no_grad():
                depth_pred = depth_pred_metric.detach().cpu().numpy().copy()
                mask_pred = torch.sigmoid(mask_logit_c).detach().cpu().numpy().copy()
                checkpoints[it] = (depth_pred, mask_pred)
                # Save predictions
                np.save(f"{out_dir}/pred_depth_{resolution_name}_iter{it}.npy", depth_pred)
                np.save(f"{out_dir}/pred_mask_{resolution_name}_iter{it}.npy", mask_pred)
                # Save visualizations
                vis_dir = os.path.join(out_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                # Depth visualization
                valid = depth_pred > 0
                vmin = depth_pred[valid].min() if valid.any() else 0
                vmax = depth_pred[valid].max() if valid.any() else 1
                norm = np.clip((depth_pred.squeeze() - vmin) / (vmax - vmin + 1e-6), 0, 1)
                colored = cm.turbo(norm)[..., :3]
                plt.imsave(os.path.join(vis_dir, f"pred_depth_{resolution_name}_iter{it}.png"), colored)
                # Mask visualization
                mask_vis = mask_pred.squeeze()
                plt.imsave(os.path.join(vis_dir, f"pred_mask_{resolution_name}_iter{it}.png"), mask_vis, cmap='gray')
                # Save model weights (encoder + decoder + optimizer)
                torch.save({
                    'iteration': it,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f"{weights_dir}/overfit_{resolution_name}_iter{it}.pt")
            print(f"  [checkpoint saved at iter {it}]")

    return checkpoints


def main():
    import os
    os.makedirs("lowres_test", exist_ok=True)

    print("Loading frame_0000...")
    rgb, depth, mask, gt_depth, gt_mask = load_frame_0000()
    print(f"Original: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}")

    alpha, beta = compute_log_params()

    # Test: Low resolution (224x224) - DINOv2 native only, center cropped
    print("\nCenter cropping to 224x224...")
    rgb_cc = center_crop(rgb, LOW_RES[0], LOW_RES[1])
    depth_cc = center_crop(depth, LOW_RES[0], LOW_RES[1])
    mask_cc = center_crop(mask, LOW_RES[0], LOW_RES[1])
    gt_depth_cc = center_crop(gt_depth, LOW_RES[0], LOW_RES[1])
    gt_mask_cc = center_crop(gt_mask, LOW_RES[0], LOW_RES[1])

    sample_cc = build_sample(rgb_cc, depth_cc, mask_cc, gt_depth_cc, gt_mask_cc, alpha, beta)
    checkpoints_cc = run_overfit("224x224", sample_cc, LOW_RES[0], LOW_RES[1], "lowres_test")

    print("\nDone! Checkpoints saved to lowres_test/")


if __name__ == "__main__":
    main()
