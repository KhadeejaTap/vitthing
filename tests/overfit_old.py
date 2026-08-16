import torch
import numpy as np
import glob
import os

from main.encoder import load_backbone, crop_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from main.decoder import DPTDecoder, denormalize_depth
from main.normalize import compute_log_params, build_input_tensor
from main.intrinsics import get_intrinsics
from main.losses import total_loss
from tests.v3_test import o
from main.losses import debug_visualize_pointcloud

NUM_ITERS = 500
ENCODER_LR = 1e-5   # rolled back from 1e-3 -- that value diverged (see run log)
DECODER_LR = 1e-4   # rolled back from 1e-2 -- that value diverged (see run log)
GRAD_CLIP_MAX_NORM = 1.0

# save a checkpoint prediction at these iterations so sharpness progression
# can be inspected in one run instead of re-running from scratch each time
CHECKPOINT_ITERS = [1, 50, 150, 300, 500]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("--- device ---")
    print(DEVICE)

    print("\n--- loading sample from hypersim_data ---")
    # Find first NPZ file in stage1/train
    npz_pattern = str(Path(__file__).resolve().parent.parent / "hypersim_data" / "stage1" / "train" / "*.npz")
    npz_files = sorted(glob.glob(npz_pattern))
    if not npz_files:
        # Fallback to val if no train files
        npz_pattern = str(Path(__file__).resolve().parent.parent / "hypersim_data" / "stage1" / "val" / "*.npz")
        npz_files = sorted(glob.glob(npz_pattern))
        if not npz_files:
            raise FileNotFoundError("No NPZ files found in stage1")

    npz_path = npz_files[0]
    print(f"Loading sample: {npz_path}")

    # Load NPZ data
    with np.load(npz_path, allow_pickle=True) as data:
        # Extract and convert fields
        rgb_np = data['rgb'].astype(np.float32)  # Already (3, H, W) ImageNet-normalized
        gt_depth_np = data['gt_depth'].astype(np.float32)  # (H, W) mm
        gt_mask_np = data['gt_mask'].astype(np.float32)  # (H, W) bool as float
        sensor_depth_np = data['sensor_depth'].astype(np.float32)  # (H, W) mm
        sensor_mask_np = data['sensor_mask'].astype(np.float32)  # (H, W) bool

    # Get dimensions - RGB is stored as (H, W, C) in NPZ
    orig_h, orig_w, orig_c = rgb_np.shape
    print(f"original resolution: {orig_h} x {orig_w} x {orig_c}")
    print(f"rgb_np shape: {rgb_np.shape}")

    # Compute log normalization parameters
    alpha, beta = compute_log_params()
    print(f"alpha={alpha:.4f} beta={beta:.4f}")

    # Create depth_input tensor (3, H, W) in [-1, 1] from sensor data
    depth_input_np = build_input_tensor(sensor_depth_np, sensor_mask_np, alpha, beta)
    print(f"depth_input_np shape before transpose: {depth_input_np.shape}")
    # build_input_tensor returns (H, W, 3), need to transpose to (3, H, W) for model
    if depth_input_np.shape[-1] == 3:  # (H, W, 3)
        depth_input_np = np.transpose(depth_input_np, (2, 0, 1))  # -> (3, H, W)
    print(f"depth_input_np shape after transpose: {depth_input_np.shape}")

    # Convert to tensors and add batch dimension
    # RGB: convert from (H, W, C) to (C, H, W) then add batch dim
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    depth_input = torch.from_numpy(depth_input_np).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    gt_depth = torch.from_numpy(gt_depth_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
    gt_mask = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
    valid_mask = torch.from_numpy(gt_mask_np).unsqueeze(0).to(DEVICE)  # (1, H, W) - squeezed version for loss
    print(f"rgb tensor shape: {rgb.shape}")
    print(f"depth_input tensor shape: {depth_input.shape}")
    print(f"gt_depth tensor shape: {gt_depth.shape}")
    print(f"gt_mask tensor shape: {gt_mask.shape}")
    print(f"valid_mask tensor shape: {valid_mask.shape}")

    # Apply cropping to multiple of 16 (patch size)
    print(f"Before cropping - rgb shape: {rgb.shape}")
    rgb = crop_to_multiple(rgb)
    depth_input = crop_to_multiple(depth_input)
    gt_depth = crop_to_multiple(gt_depth)
    gt_mask = crop_to_multiple(gt_mask)
    valid_mask = crop_to_multiple(valid_mask)
    print(f"After cropping - rgb shape: {rgb.shape}")

    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 16, out_w // 16
    print(f"cropped resolution: {out_h} x {out_w}, patch grid: {h_patch} x {w_patch}")

    # Check for invalid dimensions
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid cropped dimensions: {out_h} x {out_w}. Both must be > 0.")

    fx, fy, cx, cy = get_intrinsics(out_w, out_h)
    print(f"intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    #debug_visualize_pointcloud(gt_depth_np, gt_mask_np, fx, fy, cx, cy, out_path="pointcloud_check.html") maybe try again with flat wall
    alpha, beta = compute_log_params()
    print(f"alpha={alpha:.4f} beta={beta:.4f}")

    print("\n--- building model ---")
    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth).to(DEVICE)
    decoder = DPTDecoder().to(DEVICE)

    all_params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": decoder.parameters(), "lr": DECODER_LR},
    ])

    print(f"encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    print("\n--- overfit loop ---")
    checkpoints = {}  # iter -> (depth_pred_mm, mask_pred)

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        features = encoder(rgb, depth_input)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        # inputs and GT are already cropped to multiple of 16; use directly for loss
        depth_hat_c = depth_hat
        mask_logit_c = mask_logit

        depth_pred_metric = denormalize_depth(depth_hat_c, alpha, beta)

        loss, parts = total_loss(
            depth_pred_metric, gt_depth, gt_mask, mask_logit_c, gt_mask,
            fx, fy, cx, cy
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if it == 1 or it % 10 == 0 or it == NUM_ITERS:
            with torch.no_grad():
                valid = gt_mask.bool()
                if valid.any():
                    abs_err = (depth_pred_metric - gt_depth).abs()[valid].mean().item()
                else:
                    abs_err = float("nan")
            print(
                f"iter {it:4d}  total={loss.item():.4f}  "
                f"l1={parts['l1']:.4f}  lg={parts['lg']:.4f}  "
                f"ll={parts['ll']:.4f}  lm={parts['lm']:.4f}  "
                f"mean_abs_depth_err_mm={abs_err:.2f}"
            )

        assert not torch.isnan(loss), f"NaN loss at iter {it}"

        if it in CHECKPOINT_ITERS:
            with torch.no_grad():
                checkpoints[it] = (
                    depth_pred_metric.detach().cpu().numpy().copy(),
                    torch.sigmoid(mask_logit_c).detach().cpu().numpy().copy(),
                )
            print(f"  [checkpoint saved at iter {it}]")

    print("\ndone. if working, total loss and mean_abs_depth_err_mm should trend down,")
    print("and predictions across checkpoints should visibly sharpen over iterations.")

    for it, (depth_pred, mask_pred) in checkpoints.items():
        np.save(f"prediction_depth_mm_iter{it}.npy", depth_pred)
        np.save(f"prediction_mask_iter{it}.npy", mask_pred)
    print("saved checkpoints:", sorted(checkpoints.keys()))
    print("files: prediction_depth_mm_iter{N}.npy, prediction_mask_iter{N}.npy for N in", CHECKPOINT_ITERS)


if __name__ == "__main__":
    main()
