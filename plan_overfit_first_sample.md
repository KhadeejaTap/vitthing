# Plan to Modify tests/overfit_old.py to Overfit on First Hypersim Sample

## Goal
Modify tests/overfit_old.py to overfit on the first sample from the preprocessed hypersim_data (stage1/train/) instead of using the synthetic DToFDataset.

## Changes Needed
1. Replace DToFDataset loading with direct NPZ loading from hypersim_data
2. Load the first NPZ file from stage1/train/ directory
3. Extract and convert the required fields to proper tensor format
4. Use build_input_tensor to create depth_input from sensor data
5. Ensure all tensors have correct shapes and are on the correct device

## Current Data Flow in overfit_old.py
```python
ds = DToFDataset()  # synthetic dataset
sample = ds[0]

rgb = sample["rgb"]                    # (3, H, W)
depth_input = sample["depth_input"]    # (3, H, W) in [-1, 1] 
gt_depth = sample["gt_depth"]          # (1, H, W) in mm
gt_mask = sample["gt_mask"]            # (1, H, W) binary
valid_mask = sample["valid_mask"]      # (H, W) binary (appears duplicate of gt_mask squeezed)
```

## Target NPZ Structure (from hypersim_data)
Based on inspection of NPZ files:
- rgb: (3, H, W) float32, ImageNet-normalized
- gt_depth: (H, W) float32, depth in mm
- gt_mask: (H, W) bool, validity mask
- sensor_depth: (H, W) float32, sensor depth in mm (padded)
- sensor_mask: (H, W) bool, sensor mask (padded)
- Other metadata fields (crop_perimeter, etc.)

## Required Transformations
1. **rgb**: Already in correct format (3, H, W) ImageNet-normalized → use directly
2. **depth_input**: Need to create 3-channel tensor in [-1, 1] from sensor data:
   - Use build_input_tensor(sensor_depth, sensor_mask, alpha, beta) 
   - This should return (H, W, 3) or (3, H, W) - need to verify shape
3. **gt_depth**: (H, W) mm → unsqueeze to (1, H, W) and convert to float32
4. **gt_mask**: (H, W) bool → unsqueeze to (1, H, W) and convert to float32
5. **valid_mask**: Same as gt_mask but squeezed? Actually from DToFDataset, valid_mask is (H, W) while gt_mask is (1, H, W). We'll need to squeeze gt_mask for valid_mask.

## Implementation Steps

### 1. Import Required Functions
Add imports at top of file:
```python
from main.normalize import compute_log_params, build_input_tensor
import glob
import os
```

### 2. Replace Dataset Loading Section
Replace:
```python
print("\n--- loading sample ---")
ds = DToFDataset() # my synthetic example
sample = ds[0]

orig_h, orig_w = sample["rgb"].shape[-2], sample["rgb"].shape[-1]
print("original resolution:", orig_h, orig_w)

rgb = crop_to_multiple(sample["rgb"].unsqueeze(0)).to(DEVICE)
depth_input = crop_to_multiple(sample["depth_input"].unsqueeze(0)).to(DEVICE)
gt_depth = crop_to_multiple(sample["gt_depth"].unsqueeze(0)).to(DEVICE)
gt_mask = crop_to_multiple(sample["gt_mask"].unsqueeze(0)).to(DEVICE)
```

With:
```python
print("\n--- loading sample from hypersim_data ---")
# Find first NPZ file in stage1/train
npz_pattern = "/home/khadeeja/vitthing/hypersim_data/stage1/train/*.npz"
npz_files = sorted(glob.glob(npz_pattern))
if not npz_files:
    # Fallback to val if no train files
    npz_pattern = "/home/khadeeja/vitthing/hypersim_data/stage1/val/*.npz"
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

# Get dimensions
orig_h, orig_w = rgb_np.shape[1], rgb_np.shape[2]  # Note: rgb is (3, H, W)
print(f"original resolution: {orig_h} x {orig_w}")

# Compute log normalization parameters
alpha, beta = compute_log_params()
print(f"alpha={alpha:.4f} beta={beta:.4f}")

# Create depth_input tensor (3, H, W) in [-1, 1] from sensor data
depth_input_np = build_input_tensor(sensor_depth_np, sensor_mask_np, alpha, beta)
# Verify shape - should be (H, W, 3) or (3, H, W)
if depth_input_np.shape[-1] == 3:  # (H, W, 3)
    depth_input_np = np.transpose(depth_input_np, (2, 0, 1))  # -> (3, H, W)
# Else assume it's already (3, H, W)

# Convert to tensors and add batch dimension
rgb = torch.from_numpy(rgb_np).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
depth_input = torch.from_numpy(depth_input_np).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
gt_depth = torch.from_numpy(gt_depth_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
gt_mask = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)
valid_mask = torch.from_numpy(gt_mask_np).unsqueeze(0).to(DEVICE)  # (1, H, W) - squeezed version for loss

# Apply cropping to multiple of 16 (patch size)
rgb = crop_to_multiple(rgb)
depth_input = crop_to_multiple(depth_input)
gt_depth = crop_to_multiple(gt_depth)
gt_mask = crop_to_multiple(gt_mask)
valid_mask = crop_to_multiple(valid_mask)

out_h, out_w = rgb.shape[-2], rgb.shape[-1]
h_patch, w_patch = out_h // 16, out_w // 16
print(f"cropped resolution: {out_h} x {out_w}, patch grid: {h_patch} x {w_patch}"
```

### 3. Adjust Loss Function Call
The loss function call should remain the same since we're providing the same tensor types:
```python
loss, parts = total_loss(
    depth_pred_metric, gt_depth, gt_mask, mask_logit_c, gt_mask,  # Note: gt_mask used twice as expected
    fx, fy, cx, cy
)
```

## Verification
After making changes, run a quick test:
```bash
python3 -m tests.overfit_old
```

Check that:
1. It loads the NPZ file successfully
2. Tensor shapes are correct
3. Training begins without shape/dtype errors
4. Loss values look reasonable (not immediately NaN or extreme)

## Expected Files to Modify
- `/home/khadeeja/vitthing/tests/overfit_old.py` - main modifications

## Dependencies
- Need to ensure main/normalize.py is accessible for build_input_tensor and compute_log_params
- Need to ensure the hypersim_data preprocessing has been run and NPZ files exist

## Risk Assessment
- **Medium risk**: Changes core data loading logic
- **Mitigation**: Keep backup of original file, test with small number of iterations first
- **Fallback**: Can revert to original DToFDataset if needed