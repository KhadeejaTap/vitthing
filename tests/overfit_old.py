import torch
import numpy as np
from pathlib import Path

from main.encoder import load_backbone, crop_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from main.decoder import DPTDecoder, denormalize_depth
from main.normalize import compute_log_params
from main.preprocessed_dataset import PreprocessedHypersimDataset
from main.losses import total_loss

NUM_ITERS = 1000
ENCODER_LR = 1e-5   # rolled back from 1e-3 -- that value diverged (see run log)
DECODER_LR = 1e-4   # rolled back from 1e-2 -- that value diverged (see run log)
GRAD_CLIP_MAX_NORM = 1.0

# save a checkpoint prediction at these iterations so sharpness progression
# can be inspected in one run instead of re-running from scratch each time
CHECKPOINT_ITERS = [1, 50, 150, 300, 500, 600, 700, 900, 1000]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("--- device ---")
    print(DEVICE)

    print("\n--- loading sample from preprocessed hypersim_data ---")
    # Use PreprocessedHypersimDataset to load data
    data_dir = str(Path(__file__).resolve().parent.parent / "hypersim_data")
    dataset = PreprocessedHypersimDataset(data_dir=data_dir, stage=1, split="train")

    if len(dataset) == 0:
        raise FileNotFoundError("No NPZ files found in stage1/train")

    # Get the first sample
    sample = dataset[0]
    print(f"Loaded sample: {sample['scene']}/{sample['cam']}/frame.{sample['frame']}")

    # Extract data from sample (already formatted correctly by PreprocessedHypersimDataset)
    rgb = sample["rgb"].unsqueeze(0).to(DEVICE)                    # (1, 3, H, W) [0,1]
    depth_filled_mm = sample["depth_filled_mm"].unsqueeze(0).to(DEVICE)  # (1, 1, H, W) mm - flood-filled sensor depth
    valid_mask = sample["valid_mask"].unsqueeze(0).to(DEVICE)      # (1, H, W) bool - sparse real measurements
    gt_depth = sample["gt_depth"].unsqueeze(0).to(DEVICE)          # (1, 1, H, W) mm
    gt_mask = sample["gt_mask"].unsqueeze(0).to(DEVICE)            # (1, 1, H, W) bool
    fx = sample["fx"]
    fy = sample["fy"]
    cx = sample["cx"]
    cy = sample["cy"]

    print(f"rgb tensor shape: {rgb.shape}")
    print(f"depth_filled_mm tensor shape: {depth_filled_mm.shape}")
    print(f"valid_mask tensor shape: {valid_mask.shape}")
    print(f"gt_depth tensor shape: {gt_depth.shape}")
    print(f"gt_mask tensor shape: {gt_mask.shape}")
    print(f"intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # Compute log normalization parameters (needed for building depth_input)
    alpha, beta = compute_log_params()
    print(f"alpha={alpha:.4f} beta={beta:.4f}")

    # Build depth_input tensor for encoder: [zhat, zhat, valid_mask] -> [-1, 1]
    # where zhat = (log(depth_mm) - beta) / alpha
    depth_mm = depth_filled_mm.squeeze(1).clamp_min(1.0)  # (1, H, W)
    print(f"depth_mm shape after squeeze: {depth_mm.shape}")
    zhat = (torch.log(depth_mm) - beta) / alpha           # (1, H, W)
    print(f"zhat shape: {zhat.shape}")
    zhat = zhat.clamp(0.0, 1.0)                           # (1, H, W)
    print(f"zhat shape after clamp: {zhat.shape}")

    # Build 3-channel input tensor: [zhat, zhat, valid_mask] -> [-1, 1]
    # valid_mask already has batch dimension [1, H, W] to match zhat
    print(f"valid_mask shape: {valid_mask.shape}")
    depth_input = torch.stack([zhat, zhat, valid_mask.float()], dim=1) * 2.0 - 1.0  # (1, 3, H, W)
    print(f"depth_input tensor shape: {depth_input.shape}")

    # Apply cropping to multiple of 16 (patch size for DINOv3) - matching train_staged approach
    print(f"Before cropping - rgb shape: {rgb.shape}")
    rgb_cropped = crop_to_multiple(rgb)
    depth_input_cropped = crop_to_multiple(depth_input)
    gt_depth_cropped = crop_to_multiple(gt_depth)
    gt_mask_cropped = crop_to_multiple(gt_mask)
    print(f"After cropping - rgb shape: {rgb_cropped.shape}")

    out_h, out_w = rgb_cropped.shape[-2], rgb_cropped.shape[-1]
    h_patch, w_patch = out_h // 16, out_w // 16
    print(f"cropped resolution: {out_h} x {out_w}, patch grid: {h_patch} x {w_patch}")

    # Check for invalid dimensions
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid cropped dimensions: {out_h} x {out_w}. Both must be > 0.")

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

        features = encoder(rgb_cropped, depth_input_cropped)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        # For loss computation, inputs and GT are already cropped to multiple of 16
        depth_hat_c = depth_hat
        mask_logit_c = mask_logit
        gt_depth_c = gt_depth_cropped
        gt_mask_c = gt_mask_cropped

        depth_pred_metric = denormalize_depth(depth_hat_c, alpha, beta)

        loss, parts = total_loss(
            depth_pred_metric, gt_depth_c, gt_mask_c, mask_logit_c, gt_mask_c,
            fx, fy, cx, cy
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if it == 1 or it % 10 == 0 or it == NUM_ITERS:
            with torch.no_grad():
                valid = gt_mask_c.bool()
                if valid.any():
                    abs_err = (depth_pred_metric - gt_depth_c).abs()[valid].mean().item()
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
