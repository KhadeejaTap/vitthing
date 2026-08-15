import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedJointAttention(nn.Module):
    """
    reuses pretrained qkv/proj weights from both branches, unchanged.
    computes attention jointly over concat(image_tokens, depth_tokens).
    directional mask G = [[1,1],[0,1]] (rows=query, cols=key, order=[image,depth]):
      image query -> image key : allowed
      image query -> depth key : allowed   (depth guides image)
      depth query -> image key : BLOCKED   (image can't corrupt depth)
      depth query -> depth key : allowed
    """

    def __init__(self, img_attn, depth_attn, num_heads):
        super().__init__()
        self.img_qkv = img_attn.qkv
        self.img_proj = img_attn.proj
        self.depth_qkv = depth_attn.qkv
        self.depth_proj = depth_attn.proj
        self.num_heads = num_heads

    def _split_heads(self, x, B):
        # x: (B, N, 3C) -> q,k,v each (B, heads, N, head_dim)
        N, C3 = x.shape[1], x.shape[2]
        C = C3 // 3
        head_dim = C // self.num_heads
        qkv = x.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x_img, x_depth):
        B, N_img, C = x_img.shape
        _, N_depth, _ = x_depth.shape

        qkv_img = self.img_qkv(x_img)
        qkv_depth = self.depth_qkv(x_depth)

        q_img, k_img, v_img = self._split_heads(qkv_img, B)
        q_depth, k_depth, v_depth = self._split_heads(qkv_depth, B)

        q = torch.cat([q_img, q_depth], dim=2)  # (B, heads, N_img+N_depth, head_dim)
        k = torch.cat([k_img, k_depth], dim=2)
        v = torch.cat([v_img, v_depth], dim=2)

        N_total = N_img + N_depth
        mask = torch.zeros(N_total, N_total, device=x_img.device, dtype=q.dtype)
        mask[N_img:, :N_img] = float("-inf")  # depth queries blocked from image keys

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N_total, C)

        out_img = self.img_proj(out[:, :N_img, :])
        out_depth = self.depth_proj(out[:, N_img:, :])
        return out_img, out_depth


class DualBranchFusionBlock(nn.Module):
    """
    one fused transformer layer: reuses norm1/mlp/norm2/layerscale from each
    branch's pretrained block untouched, only the attention op is joint+masked.
    """

    def __init__(self, rgb_block, depth_block, num_heads):
        super().__init__()
        self.rgb_block = rgb_block
        self.depth_block = depth_block
        self.joint_attn = MaskedJointAttention(rgb_block.attn, depth_block.attn, num_heads)

    def forward(self, x_img, x_depth):
        img_normed = self.rgb_block.norm1(x_img)
        depth_normed = self.depth_block.norm1(x_depth)

        attn_img, attn_depth = self.joint_attn(img_normed, depth_normed)

        x_img = x_img + self.rgb_block.ls1(attn_img)
        x_depth = x_depth + self.depth_block.ls1(attn_depth)

        x_img = x_img + self.rgb_block.ls2(self.rgb_block.mlp(self.rgb_block.norm2(x_img)))
        x_depth = x_depth + self.depth_block.ls2(self.depth_block.mlp(self.depth_block.norm2(x_depth)))

        return x_img, x_depth


class DualBranchEncoder(nn.Module):
    """
    wraps two pretrained dinov2 backbones, fuses every layer via masked joint
    attention, all 12 layers (no dynamic token sampling yet - fixed full grid).
    returns intermediate outputs at layers 6, 12 for the decoder.
    """

    def __init__(self, model_rgb, model_depth, num_heads=6, extract_layers=(6, 12)):
        super().__init__()
        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.extract_layers = extract_layers

        assert len(model_rgb.blocks) == len(model_depth.blocks)
        self.fusion_blocks = nn.ModuleList([
            DualBranchFusionBlock(model_rgb.blocks[i], model_depth.blocks[i], num_heads)
            for i in range(len(model_rgb.blocks))
        ])

    def forward(self, rgb, depth_input):
        # patch embed + cls/reg tokens + pos embed, each branch's own pretrained layers
        x_img = self.model_rgb.prepare_tokens_with_masks(rgb)
        x_depth = self.model_depth.prepare_tokens_with_masks(depth_input)

        features = {}
        for i, block in enumerate(self.fusion_blocks, start=1):
            x_img, x_depth = block(x_img, x_depth)
            if i in self.extract_layers:
                features[i] = (x_img.clone(), x_depth.clone())

        return features


if __name__ == "__main__":
    from encoder import load_backbone, pad_to_multiple
    from code.dataset import DToFDataset

    model_rgb = load_backbone()
    model_depth = load_backbone()

    ds = DToFDataset()
    sample = ds[0]

    rgb = pad_to_multiple(sample["rgb"].unsqueeze(0))
    depth_input = pad_to_multiple(sample["depth_input"].unsqueeze(0))

    print("--- input shapes ---")
    print("rgb:", rgb.shape)
    print("depth_input:", depth_input.shape)

    encoder = DualBranchEncoder(model_rgb, model_depth)

    # --- check 1: mask correctness (values, not just existence) ---
    print("\n--- mask check ---")
    dummy_img = torch.randn(1, 5, 384)
    dummy_depth = torch.randn(1, 3, 384)
    ja = encoder.fusion_blocks[0].joint_attn
    N_img, N_depth = dummy_img.shape[1], dummy_depth.shape[1]
    N_total = N_img + N_depth
    mask = torch.zeros(N_total, N_total)
    mask[N_img:, :N_img] = float("-inf")
    print(mask)
    assert torch.all(mask[:N_img, :] == 0), "image queries should be unrestricted"
    assert torch.all(mask[N_img:, :N_img] == float("-inf")), "depth->image must be blocked"
    assert torch.all(mask[N_img:, N_img:] == 0), "depth->depth should be unrestricted"
    print("mask values correct")

    # --- check 2: forward pass, real data ---
    print("\n--- forward pass ---")
    features = encoder(rgb, depth_input)
    for layer_idx, (feat_img, feat_depth) in features.items():
        print(f"layer {layer_idx}: img {feat_img.shape}, depth {feat_depth.shape}")
        assert not torch.isnan(feat_img).any(), f"NaN in img features at layer {layer_idx}"
        assert not torch.isnan(feat_depth).any(), f"NaN in depth features at layer {layer_idx}"
        assert not torch.isinf(feat_img).any(), f"Inf in img features at layer {layer_idx}"
        assert not torch.isinf(feat_depth).any(), f"Inf in depth features at layer {layer_idx}"
    print("no NaN/Inf in outputs")

    # --- check 3: real correctness test ---
    # depth branch must be BLIND to image tokens. if we change rgb input
    # but keep depth input fixed, depth-branch output must stay identical.
    print("\n--- depth-blind-to-image test ---")
    rgb_alt = torch.randn_like(rgb)  # completely different image input

    features_a = encoder(rgb, depth_input)
    features_b = encoder(rgb_alt, depth_input)

    for layer_idx in features_a:
        img_a, depth_a = features_a[layer_idx]
        img_b, depth_b = features_b[layer_idx]

        img_diff = (img_a - img_b).abs().max().item()
        depth_diff = (depth_a - depth_b).abs().max().item()

        print(f"layer {layer_idx}: img_diff={img_diff:.6f}  depth_diff={depth_diff:.8f}")
        assert depth_diff < 1e-5, (
            f"FAIL layer {layer_idx}: depth branch changed when only rgb input changed "
            f"(diff={depth_diff}) -- mask direction is wrong, depth is seeing image tokens"
        )
        assert img_diff > 1e-3, (
            f"WARN layer {layer_idx}: img branch barely changed with different rgb input, "
            f"check img branch is actually using rgb input"
        )
    print("PASSED: depth branch unaffected by image changes, img branch is affected")
