#!/usr/bin/env python3
"""
Random weights overfit test - compares pretrained vs random DINOv2 initialization.
Run this to see if pretrained weights are the bottleneck.
"""
import torch
import numpy as np
from PIL import Image

from encoder import load_backbone, pad_to_multiple, crop_to_original
from fusion import DualBranchEncoder
from decoder import DPTDecoder, denormalize_depth
from normalize import compute_log_params, build_input_tensor
from intrinsics import get_intrinsics
from losses import total_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ITERS = 300
ENCODER_LR = 1e-4
DECODER_LR = 1e-3
GRAD_CLIP_MAX_NORM = 1.0
CHECKPOINT_ITERS = [1, 50, 150, 300]


def load_frame_0000():
    rgb = np.array(Image.open('data/frame_0000_rgb.png')).astype(np.float32) / 255.0
    depth = np.load('data/frame_0000_depth_proj_mm.npy').astype(np.float32)
    mask = np.load('data/frame_0000_proj_valid_mask.npy').astype(np.float32)
    gt_depth = np.load('data/frame_0000_gt_mm.npy').astype(np.float32)
    gt_mask = np.load('data/frame_0000_gt_mask.npy').astype(np.float32)
    return rgb, depth, mask, gt_depth, gt_mask


def build_sample(rgb, depth, mask, gt_depth, gt_mask, alpha, beta):
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


def run_overfit(name, init_mode, sample, orig_h, orig_w, out_dir):
    print(f"\n{'='*60}")
    print(f"  OVERFIT TEST: {name} (init_mode={init_mode})")
    print(f"{'='*60}")

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0)).to(DEVICE)
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0)).to(DEVICE)
    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 14, out_w // 14
    print(f"Padded: {out_h}x{out_w}, Patch grid: {h_patch}x{w_patch}")

    gt_depth = sample["gt_depth"].unsqueeze(0).to(DEVICE)
    gt_mask = sample["gt_mask"].unsqueeze(0).to(DEVICE)

    fx, fy, cx, cy = get_intrinsics()
    alpha, beta = sample["alpha"], sample["beta"]

    from encoder import reset_shared_random
    reset_shared_random()  # Ensure fresh shared state per run

    model_rgb = load_backbone(init_mode=init_mode)
    model_depth = load_backbone(init_mode=init_mode)
    encoder = DualBranchEncoder(model_rgb, model_depth).to(DEVICE)
    decoder = DPTDecoder().to(DEVICE)

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": decoder.parameters(), "lr": DECODER_LR},
    ])

    checkpoints = {}

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        features = encoder(rgb, depth_input)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        depth_hat_c = crop_to_original(depth_hat, orig_h, orig_w)
        mask_logit_c = crop_to_original(mask_logit, orig_h, orig_w)

        depth_pred_metric = denormalize_depth(depth_hat_c, alpha, beta)

        loss, parts = total_loss(
            depth_pred_metric, gt_depth, gt_mask, mask_logit_c, gt_mask,
            fx, fy, cx, cy
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if it == 1 or it % 50 == 0 or it == NUM_ITERS:
            with torch.no_grad():
                valid = gt_mask.bool()
                if valid.any():
                    abs_err = (depth_pred_metric - gt_depth).abs()[valid].mean().item()
                else:
                    abs_err = float("nan")
            print(f"iter {it:4d}  total={loss.item():.4f}  l1={parts['l1']:.4f}  mean_abs_err={abs_err:.2f}mm")

        if torch.isnan(loss):
            print(f"  NaN at iter {it}! Breaking.")
            break

        if it in CHECKPOINT_ITERS:
            with torch.no_grad():
                depth_pred = depth_pred_metric.detach().cpu().numpy().copy()
                mask_pred = torch.sigmoid(mask_logit_c).detach().cpu().numpy().copy()
                checkpoints[it] = (depth_pred, mask_pred)
                np.save(f"{out_dir}/pred_depth_{name}_iter{it}.npy", depth_pred)
                np.save(f"{out_dir}/pred_mask_{name}_iter{it}.npy", mask_pred)
            print(f"  [checkpoint saved at iter {it}]")

    return checkpoints


def main():
    import os
    os.makedirs("random_weights_test", exist_ok=True)

    print("Loading frame_0000...")
    rgb, depth, mask, gt_depth, gt_mask = load_frame_0000()
    print(f"Original: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}")

    alpha, beta = compute_log_params()
    sample = build_sample(rgb, depth, mask, gt_depth, gt_mask, alpha, beta)

    # Test: shared_random init (random but identical for RGB & depth)
    run_overfit("shared_random", "shared_random", sample, 540, 960, "random_weights_test")

    print("\nDone! Checkpoints in random_weights_test/")
    print("Compare with pretrained baseline in resize+conv+maxdim+l1only/")


if __name__ == "__main__":
    main()