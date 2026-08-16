import torch
import torch.nn.functional as F

WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / "dinov3-weights" / "dinov3_vits16.pth")

# Global cache for shared_random init (reversible, no persistent state)
_SHARED_RANDOM_STATE = None


def load_backbone(pretrained=True, init_mode="pretrained"):
    """
    init_mode:
      - "pretrained": load DINOv3 weights (default, stable)
      - "random": fully random init (unstable)
      - "shared_random": random but identical weights for RGB & depth branches
    """
    global _SHARED_RANDOM_STATE
    model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)

    if init_mode == "pretrained":
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif init_mode == "shared_random":
        if _SHARED_RANDOM_STATE is None:
            _SHARED_RANDOM_STATE = model.state_dict()
        model.load_state_dict(_SHARED_RANDOM_STATE, strict=True)
    # "random" does nothing - keeps torch.hub's random init

    return model


def reset_shared_random():
    """Clear shared random state (for reproducibility across runs)."""
    global _SHARED_RANDOM_STATE
    _SHARED_RANDOM_STATE = None


def crop_to_multiple(x, patch=16):
    """Crop symmetrically to the largest multiple of patch size.
    Prints a message if cropping occurs."""
    h, w = x.shape[-2:]
    target_h = (h // patch) * patch
    target_w = (w // patch) * patch

    if target_h != h or target_w != w:
        # Compute how much to crop from each side (symmetric)
        crop_h = h - target_h
        crop_w = w - target_w
        crop_top = crop_h // 2
        crop_bottom = crop_h - crop_top
        crop_left = crop_w // 2
        crop_right = crop_w - crop_left
        print(f"[crop_to_multiple] Cropping from ({h},{w}) to ({target_h},{target_w}) "
              f"(top:{crop_top}, bottom:{crop_bottom}, left:{crop_left}, right:{crop_right})")
        x = x[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]

    return x


def crop_to_original(x, orig_h, orig_w, patch=16):
    """After cropping inputs to multiple of patch size, we do not pad back.
    This function is kept for compatibility but does nothing (returns x)."""
    return x


if __name__ == "__main__":
    from main.dataset import DToFDataset

    model_rgb = load_backbone()
    model_depth = load_backbone()

    ds = DToFDataset()
    sample = ds[0]

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0))          # (1,3,H,W)
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0))

    print("rgb in:", rgb.shape)
    print("depth in:", depth_input.shape)

    out_rgb = model_rgb.forward_features(rgb)
    out_depth = model_depth.forward_features(depth_input)

    print("rgb out:", {k: v.shape for k, v in out_rgb.items() if v is not None})
    print("depth out:", {k: v.shape for k, v in out_depth.items() if v is not None})

    print(model_rgb.blocks[0].attn)
    print(model_rgb.blocks[0].attn.qkv.weight.shape)
