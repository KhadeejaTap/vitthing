import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# UNCERTAINTIES (paper does not fully specify these, flagged clearly):
#
# U1. How rgb-branch and depth-branch features (two separate tensors
#     per extraction layer, from fusion.py) get combined before the
#     decoder. Paper doesn't say. I concat channel-wise then 1x1 conv
#     to fuse -- simple, cheap, standard choice, but not confirmed
#     paper's actual method.
#
# U2. ViT layers don't change spatial resolution (unlike CNN backbones
#     DPT was originally designed for), so "multi-scale" here comes
#     entirely from decoder-side upsampling, not from different-res
#     ViT features. I treat layer12 (deepest) as the coarsest/first
#     decoder stage and layer6 as a mid-level skip connection fused in
#     partway through upsampling. This mirrors standard DPT/MoGe-2
#     practice but the paper doesn't spell out which ViT layer maps to
#     which decoder stage.
#
# U3. Exact fusion-block internals (residual conv design inside each
#     RefineNet-style block) aren't given. Using a standard lightweight
#     residual conv block (matches typical MoGe/DPT decoder design).
#
# U4. Patch size 14 doesn't divide evenly into powers of 2 relative to
#     input resolution, so exact upsampling factors per stage won't
#     land exactly on the input's H,W. I do progressive 2x upsamples
#     through the 5 channel stages, then a final bilinear resize to
#     the exact target resolution. This final-resize step is confirmed
#     against MoGe-2 (arXiv:2507.02546, Appendix A.1): "the output map
#     is resized through bilinear interpolation to match the raw image
#     size" -- so keeping bilinear here specifically is correct, not a
#     workaround.
#
# U5. RESOLVED (was previously plain bilinear, then ConvTranspose2d):
#     per-stage upsampling inside RefineStage is now resize+conv
#     (bilinear upsample + 3x3 conv), avoiding checkerboard artifacts
#     from kernel=2 stride=2 transpose conv while keeping learnable
#     upsampling. This is the standard DPT/MiDaS approach and is
#     compatible with "following MoGe-2" since MoGe-2's exact upsampling
#     is not paper-verbatim. The final bilinear resize to exact target
#     resolution (U4) is retained per MoGe-2 Appendix A.1.
#
# U6. Still open: whether layer6 and layer12 features should be *summed*
#     into one shared starting feature (MoGe-2's approach for its 4
#     extracted layers) vs. treated asymmetrically as main-input +
#     skip-connection (current design, closer to classic DPT). The dToF
#     paper calls its decoder both "DPT-style" and "following MoGe-2" in
#     the same sentence, which doesn't resolve this. Left as skip-connection
#     for now -- flagging in case blur persists after this fix, since this
#     is the next most likely architectural culprit.
# ============================================================


def reassemble(tokens, num_cls, num_reg, h_patch, w_patch):
    """
    (B, N_total, C) -> (B, C, h_patch, w_patch)
    drops cls + register tokens (non-spatial), reshapes patch tokens
    back into a 2D grid.
    """
    B, N, C = tokens.shape
    patch_tokens = tokens[:, num_cls + num_reg:, :]  # drop cls+reg
    assert patch_tokens.shape[1] == h_patch * w_patch, (
        f"token count {patch_tokens.shape[1]} != h*w {h_patch*w_patch}"
    )
    x = patch_tokens.permute(0, 2, 1).reshape(B, C, h_patch, w_patch)
    return x


class BranchFusion(nn.Module):
    """U1: combine rgb-branch + depth-branch features at one extraction layer."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1)

    def forward(self, feat_img, feat_depth):
        x = torch.cat([feat_img, feat_depth], dim=1)  # channel concat
        return self.proj(x)


class ResidualConvBlock(nn.Module):
    """U3: lightweight residual conv block used inside each refine stage."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class RefineStage(nn.Module):
    """
    upsample previous stage 2x AND reduce channels to this stage's target
    width. Uses resize+conv (bilinear upsample + 3x3 conv) instead of
    ConvTranspose2d to avoid checkerboard artifacts from kernel=2, stride=2.
    Still learnable, still progressive 2x upsampling per stage.
    Optionally fuses a skip connection from the encoder neck after upsampling.
    """

    def __init__(self, in_ch, out_ch, has_skip):
        super().__init__()
        self.has_skip = has_skip
        if has_skip:
            self.skip_proj = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.refine = ResidualConvBlock(out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)  # learned 2x upsample + channel reduction in one op
        if self.has_skip and skip is not None:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip)
        x = self.refine(x)
        return x


class DPTDecoder(nn.Module):
    """
    consumes fused layer6, 12 features (from encoder, already
    combined across rgb/depth branches via BranchFusion), progressively
    upsamples through channel schedule 384 -> 256 -> 64 -> 32 -> 16,
    outputs depth head (1ch, normalized) and mask head (1ch, logit).
    """

    CHANNELS = [384, 256, 64, 32, 16]  # paper's original channels

    def __init__(self, embed_dim=384, num_cls=1, num_reg=4):
        super().__init__()
        self.num_cls = num_cls
        self.num_reg = num_reg

        # Fuse each extracted layer to corresponding decoder stage channels
        self.fuse_l12 = BranchFusion(embed_dim, self.CHANNELS[0])  # layer12 -> stage0 input (384)
        self.fuse_l6 = BranchFusion(embed_dim, self.CHANNELS[2])   # layer6 -> skip at stage2 (64)

        # stage0: 384 (from l12) -> 256, no skip
        # stage1: 256 -> 64, fuses l6 skip
        # stage2: 64 -> 32, no skip
        # stage3: 32 -> 16, no skip
        self.stage0 = RefineStage(self.CHANNELS[0], self.CHANNELS[1], has_skip=False)
        self.stage1 = RefineStage(self.CHANNELS[1], self.CHANNELS[2], has_skip=True)
        self.stage2 = RefineStage(self.CHANNELS[2], self.CHANNELS[3], has_skip=False)
        self.stage3 = RefineStage(self.CHANNELS[3], self.CHANNELS[4], has_skip=False)

        self.depth_head = nn.Conv2d(self.CHANNELS[4], 1, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(self.CHANNELS[4], 1, kernel_size=3, padding=1)

        # Initialize depth head to output reasonable D_hat for typical depth (~3000mm)
        # D_hat = (log(D) - beta) / alpha ≈ (log(3000) - 5.7038) / 3.3242 ≈ 0.69
        nn.init.zeros_(self.depth_head.weight)
        nn.init.constant_(self.depth_head.bias, 0.69)
        # Initialize mask head to predict valid (positive logit)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, features, h_patch, w_patch, out_h, out_w):
        """
        features: dict from DualBranchEncoder, {6: (img_tok, depth_tok), 12: (...)}
        h_patch, w_patch: patch grid size (e.g. 39, 69)
        out_h, out_w: target output resolution (padded input size)
        """
        img6, depth6 = features[6]
        img12, depth12 = features[12]

        feat6_img = reassemble(img6, self.num_cls, self.num_reg, h_patch, w_patch)
        feat6_depth = reassemble(depth6, self.num_cls, self.num_reg, h_patch, w_patch)
        feat12_img = reassemble(img12, self.num_cls, self.num_reg, h_patch, w_patch)
        feat12_depth = reassemble(depth12, self.num_cls, self.num_reg, h_patch, w_patch)

        skip_l6 = self.fuse_l6(feat6_img, feat6_depth)     # (B,64,h,w)
        x = self.fuse_l12(feat12_img, feat12_depth)        # (B,384,h,w)

        x = self.stage0(x)              # (B,256, 2h, 2w)
        x = self.stage1(x, skip=skip_l6)  # (B,64,  4h, 4w)
        x = self.stage2(x)              # (B,32,  8h, 8w)
        x = self.stage3(x)              # (B,16, 16h,16w)

        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)

        depth_hat = self.depth_head(x)   # normalized depth D-hat, unconstrained range
        mask_logit = self.mask_head(x)   # validity logit, sigmoid at loss time
        return depth_hat, mask_logit


def denormalize_depth(depth_hat, alpha, beta):
    """eq(3): D = exp(alpha * D_hat + beta)"""
    return torch.exp(alpha * depth_hat + beta)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from encoder import load_backbone, pad_to_multiple
    from main.dataset import DToFDataset
    from fusion import DualBranchEncoder
    from normalize import compute_log_params

    model_rgb = load_backbone()
    model_depth = load_backbone()

    ds = DToFDataset()
    sample = ds[0]

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0))
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0))
    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 14, out_w // 14

    print("h_patch, w_patch:", h_patch, w_patch)

    encoder = DualBranchEncoder(model_rgb, model_depth)
    features = encoder(rgb, depth_input)

    decoder = DPTDecoder()
    depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

    print("depth_hat:", depth_hat.shape, "range:", depth_hat.min().item(), depth_hat.max().item())
    print("mask_logit:", mask_logit.shape)

    alpha, beta = compute_log_params()
    depth_metric = denormalize_depth(depth_hat, alpha, beta)
    print("depth_metric (mm):", depth_metric.shape, "range:", depth_metric.min().item(), depth_metric.max().item())

    assert not torch.isnan(depth_hat).any(), "NaN in depth_hat"
    assert not torch.isnan(mask_logit).any(), "NaN in mask_logit"
    assert depth_hat.shape[-2:] == (out_h, out_w), "depth_hat resolution mismatch"
    assert mask_logit.shape[-2:] == (out_h, out_w), "mask_logit resolution mismatch"
    print("PASSED: shapes and values sane")
