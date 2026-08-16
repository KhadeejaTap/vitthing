FX = FY = 886.81 * 0.25  # = 221.7025

def get_intrinsics(crop_w, crop_h):
    return FX, FY, crop_w / 2, crop_h / 2

if __name__ == "__main__":
    fx, fy, cx, cy = get_intrinsics()
    print(f"fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
