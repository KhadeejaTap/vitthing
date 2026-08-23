#!/usr/bin/env python3
"""Test script to verify the loss function fixes"""

import torch
import sys
import os
sys.path.append('/home/khadeeja/vitthing')

from main.losses import (
    depth_weighted_l1_loss,
    global_scale_invariant_loss,
    local_scale_invariant_loss,
    mask_bce_loss,
    total_loss
)

def test_loss_functions():
    print("Testing loss functions...")

    # Create dummy data
    B, H, W = 2, 32, 32
    fx, fy, cx, cy = 100.0, 100.0, 16.0, 16.0

    pred_depth = torch.rand(B, 1, H, W) * 5000 + 300
    gt_depth = torch.rand(B, 1, H, W) * 5000 + 300
    gt_mask = (torch.rand(B, 1, H, W) > 0.2).float()
    mask_logit = torch.randn(B, 1, H, W)
    mask_gt = (torch.rand(B, 1, H, W) > 0.2).float()

    # Test individual losses
    print("\n1. Testing depth_weighted_l1_loss:")
    l1_loss, l1_misc = depth_weighted_l1_loss(pred_depth, gt_depth, gt_mask)
    print(f"   Loss type: {type(l1_loss)}")
    print(f"   Loss value: {l1_loss.item()}")
    print(f"   Is tensor: {torch.is_tensor(l1_loss)}")

    print("\n2. Testing global_scale_invariant_loss:")
    lg_loss, lg_misc = global_scale_invariant_loss(pred_depth, gt_depth, gt_mask, fx, fy, cx, cy)
    print(f"   Loss type: {type(lg_loss)}")
    print(f"   Loss value: {lg_loss.item()}")
    print(f"   Is tensor: {torch.is_tensor(lg_loss)}")

    print("\n3. Testing local_scale_invariant_loss:")
    ll_loss, ll_misc = local_scale_invariant_loss(pred_depth, gt_depth, gt_mask, fx, fy, cx, cy)
    print(f"   Loss type: {type(ll_loss)}")
    print(f"   Loss value: {ll_loss.item()}")
    print(f"   Is tensor: {torch.is_tensor(ll_loss)}")

    print("\n4. Testing mask_bce_loss:")
    lm_loss, lm_misc = mask_bce_loss(mask_logit, mask_gt)
    print(f"   Loss type: {type(lm_loss)}")
    print(f"   Loss value: {lm_loss.item()}")
    print(f"   Is tensor: {torch.is_tensor(lm_loss)}")

    print("\n5. Testing total_loss:")
    total, misc = total_loss(pred_depth, gt_depth, gt_mask, mask_logit, mask_gt, fx, fy, cx, cy)
    print(f"   Loss type: {type(total)}")
    print(f"   Loss value: {total.item()}")
    print(f"   Is tensor: {torch.is_tensor(total)}")

    # Test with some empty batches to verify valid_batches logic
    print("\n6. Testing with empty batches:")
    empty_gt_mask = torch.zeros_like(gt_mask)  # All invalid
    l1_loss_empty, _ = depth_weighted_l1_loss(pred_depth, gt_depth, empty_gt_mask)
    lg_loss_empty, _ = global_scale_invariant_loss(pred_depth, gt_depth, empty_gt_mask, fx, fy, cx, cy)
    ll_loss_empty, _ = local_scale_invariant_loss(pred_depth, gt_depth, empty_gt_mask, fx, fy, cx, cy)
    print(f"   L1 loss with empty mask: {l1_loss_empty.item()}")
    print(f"   LG loss with empty mask: {lg_loss_empty.item()}")
    print(f"   LL loss with empty mask: {ll_loss_empty.item()}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_loss_functions()