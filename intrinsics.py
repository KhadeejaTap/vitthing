import math

# synthetic camera, known values
W_ORIG, H_ORIG = 960, 540
HFOV_DEG = 93.012973

# padding applied by encoder.pad_to_multiple (960x540 -> 966x546, symmetric split)
LEFT_PAD, TOP_PAD = 3, 3


def get_intrinsics():
    """returns fx, fy, cx, cy for the PADDED rgb resolution (966x546)."""
    hfov_rad = math.radians(HFOV_DEG)
    fx = (W_ORIG / 2) / math.tan(hfov_rad / 2)
    fy = fx  # assumes square pixels (standard for synthetic renders)
    cx = W_ORIG / 2 + LEFT_PAD
    cy = H_ORIG / 2 + TOP_PAD
    return fx, fy, cx, cy


def get_intrinsics_for_res(orig_h, orig_w, left_pad=3, top_pad=3):
    """returns fx, fy, cx, cy for a given original resolution (before padding).
    Computes fx/fy from HFOV, then adds padding offset to cx/cy."""
    hfov_rad = math.radians(HFOV_DEG)
    fx = (orig_w / 2) / math.tan(hfov_rad / 2)
    fy = fx
    cx = orig_w / 2 + left_pad
    cy = orig_h / 2 + top_pad
    return fx, fy, cx, cy


if __name__ == "__main__":
    fx, fy, cx, cy = get_intrinsics()
    print(f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
