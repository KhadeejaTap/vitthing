import glob
import os
import re
import numpy as np

GT_GLOB = "data/frame_*_gt_mm.npy"


def _extract_idx(path):
    m = re.search(r"frame_(\d+)_gt_mm\.npy", path)
    return m.group(1)


def make_gt_mask(depth_gt):
    """valid where depth is a real positive finite measurement."""
    return (np.isfinite(depth_gt) & (depth_gt > 0)).astype(np.float32)


if __name__ == "__main__":
    files = sorted(glob.glob(GT_GLOB))
    print(f"found {len(files)} gt frames")

    for f in files:
        idx = _extract_idx(f)
        depth_gt = np.load(f)

        mask = make_gt_mask(depth_gt)

        out_path = os.path.join("data", f"frame_{idx}_gt_mask.npy")
        np.save(out_path, mask)

        valid_pct = mask.mean() * 100
        print(f"{f} -> {out_path}  ({valid_pct:.1f}% valid)")

    print("done")
