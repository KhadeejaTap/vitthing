Plan to change pad_to_multiple to crop to multiple of 16 and adjust training loop accordingly.

## Changes in encoder.py

1. pad_to_multiple(x, patch=16):
   - Compute target height and width as the largest multiple of patch <= current size.
   - If current size is not target size, print a message indicating cropping (showing original and target size).
   - Perform symmetric crop (remove equal amounts from both sides; if odd, extra removed from bottom/right).
   - Return the cropped tensor.

2. crop_to_original(x, orig_h, orig_w, patch=16):
   - Change docstring to indicate that this function is now a no-op because we are cropping inputs and GT to the same size and not padding back for loss computation.
   - Alternatively, we can keep it as a no-op (return x) and update the docstring.

## Changes in overfit_old.py (and similar training scripts)

We will adjust the training loop to:
   - Crop the input rgb and depth_input using pad_to_multiple (which now crops).
   - Also crop the gt_depth and gt_mask to the same size using pad_to_multiple.
   - Remove the crop_to_original calls for the model output because we are already at the cropped resolution.
   - Use the cropped tensors directly for model input and loss computation.

Example diff for overfit_old.py:

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0)).to(DEVICE)
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0)).to(DEVICE)
    gt_depth = pad_to_multiple(sample["gt_depth"].unsqueeze(0)).to(DEVICE)
    gt_mask = pad_to_multiple(sample["gt_mask"].unsqueeze(0)).to(DEVICE)

    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 16, out_w // 16

    # ... model forward ...

    # No need to crop back: depth_hat and mask_logit are already at (out_h, out_w)
    depth_hat_c = depth_hat
    mask_logit_c = mask_logit

    # ... loss computation ...

## Notes

- The intrinsics in intrinsics.py currently assume a fixed padding (3,3) for the original 960x540 resolution. After cropping, the effective principal point shifts. The user may need to adjust intrinsics.py accordingly if the shift affects accuracy. However, the user's request was focused on the cropping mechanism, so we leave intrinsics adjustment to the user if needed.
- Other files that use pad_to_multiple and crop_to_original (e.g., test.py, train.py) may need similar adjustments. We will start with overfit_old.py and then update other files if the user confirms.