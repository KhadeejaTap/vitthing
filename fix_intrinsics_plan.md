# Plan to Fix Intrinsics Calculation for Arbitrary Resolutions

## Problem
The current `get_intrinsics()` function in `intrinsics.py` is hardcoded for a specific synthetic camera resolution (960x540 + padding = 966x546). When used with downsampled images (e.g., 256x144), the intrinsics are incorrect, leading to wrong geometry calculations in the loss functions.

## Solution
Modify the intrinsics calculation to work for arbitrary resolutions while maintaining the same camera model (93-degree HFOV, square pixels).

## Approach
Instead of scaling from a fixed resolution, compute the intrinsics directly for the target resolution using the same camera parameters:
- Same HFOV (93.012973 degrees)
- Square pixels (fx = fy)
- Principal point at image center (cx = width/2, cy = height/2)

## Changes Needed

### 1. Modify intrinsics.py to add a flexible function
Keep the existing `get_intrinsics()` for backward compatibility, but add a new function that takes resolution parameters:

```python
def get_intrinsics_for_resolution(width, height, hfov_deg=93.012973):
    """
    Compute camera intrinsics for a given resolution using the synthetic camera model.
    
    Args:
        width: image width in pixels
        height: image height in pixels
        hfov_deg: horizontal field of view in degrees (default: 93.012973 from synthetic camera)
    
    Returns:
        fx, fy, cx, cy: intrinsics for the specified resolution
    """
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2) / math.tan(hfov_rad / 2)
    fy = fx  # square pixels
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy
```

### 2. Update tests/overfit_old.py to use the new function
Replace the current intrinsics loading and scaling logic with a direct call to the resolution-based function:

```python
# Get intrinsics for current image resolution
fx, fy, cx, cy = get_intrinsics_for_resolution(out_w, out_h)
print(f"intrinsics (for {out_w}x{out_h}): fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

alpha, beta = compute_log_params()
print(f"alpha={alpha:.4f} beta={beta:.4f}")
```

## Why This Approach
1. **Correct Geometry**: The intrinsics will match the actual image being used
2. **Same Camera Model**: Maintains the 93-degree HFOV and square pixel assumption from the original synthetic camera
3. **Simple and Direct**: No need for scaling factors or padding compensation
4. **Flexible**: Works for any resolution without modification

## Verification
For the 256x144 image we saw in the debug output:
- Expected cx ≈ 256/2 = 128
- Expected cy ≈ 144/2 = 72
- fx and fy will be computed based on 93-degree HFOV

This should produce reasonable values that match the image geometry, fixing the issue where cx=483, cy=273 were inappropriate for a 256x144 image.

## Risk Assessment
- Low risk: Only changes how intrinsics are computed
- Easy to verify: Check that printed intrinsics match half the width/height for cx/cy
- Maintains compatibility: Existing `get_intrinsics()` unchanged
- Physically plausible: Uses same camera model as original synthetic data