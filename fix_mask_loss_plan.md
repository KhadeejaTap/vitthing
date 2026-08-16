# Plan to Fix Mask Loss Computation Bug

## Problem
In `main/losses.py`, the `mask_bce_loss` function (lines 341-346) incorrectly returns a hardcoded `{"lm": 0.0}` instead of the actual computed loss value. This causes:
- Mask loss to always show as 0.0000 in training output
- No meaningful gradient signal for the mask prediction branch
- The mask branch fails to learn proper validity predictions

## Location
File: `/home/khadeeja/vitthing/main/losses.py`
Function: `mask_bce_loss` (lines 341-346)

## Current Code
```python
def mask_bce_loss(mask_logit: torch.Tensor, mask_gt: torch.Tensor):
    """
    Eq. 7: Binary cross-entropy for validity mask.
    Lm = -sum_i [m_i * log(m̃_i) + (1-m_i) * log(1-m̃_i)]
    """
    return F.binary_cross_entropy_with_logits(mask_logit, mask_gt), {"lm": 0.0}
```

## Fix
Change the function to return the actual loss value in the metrics dictionary:

```python
def mask_bce_loss(mask_logit: torch.Tensor, mask_gt: torch.Tensor):
    """
    Eq. 7: Binary cross-entropy for validity mask.
    Lm = -sum_i [m_i * log(m̃_i) + (1-m_i) * log(1-m̃_i)]
    """
    lm_loss = F.binary_cross_entropy_with_logits(mask_logit, mask_gt)
    return lm_loss, {"lm": lm_loss.item()}
```

## Verification Steps
1. Apply the fix to `main/losses.py`
2. Run a quick test to verify the mask loss returns non-zero values:
   ```python
   import torch
   import torch.nn.functional as F
   from main.losses import mask_bce_loss
   
   # Test with random data
   logits = torch.randn(2, 1, 10, 10, requires_grad=True)
   targets = (torch.rand(2, 1, 10, 10) > 0.5).float()
   
   loss, misc = mask_bce_loss(logits, targets)
   print(f"Loss: {loss.item()}")
   print(f"Metrics: {misc}")
   
   # Verify loss is not zero and metrics match
   assert loss.item() != 0.0, "Loss should not be zero"
   assert misc["lm"] == loss.item(), "Metrics should match loss value"
   assert not torch.isnan(loss), "Loss should not be NaN"
   print("✓ Mask loss fix verified")
   ```
3. Run the overfitting script briefly to confirm mask loss now shows meaningful values:
   ```bash
   python3 -m tests.overfit_old
   ```
   (Monitor for lm values > 0.0000 in the output)

## Expected Outcome
After the fix:
- Training output will show meaningful lm values (e.g., lm=0.1234) instead of lm=0.0000
- The mask branch will receive proper gradient signals for learning
- Overall model performance may improve as both depth and mask branches learn properly

## Risk Assessment
- **Risk**: Very low - this is a simple correction to return the correct value
- **Impact**: High - fixes a critical learning issue in the mask prediction branch
- **Testing**: Easy to verify with unit tests and brief training runs