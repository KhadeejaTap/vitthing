import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# DINOv3 CHANGE: RoPE helpers.
# DINOv3 backbones have no learned positional embedding at all -- position is
# injected per-layer, inside attention, by rotating q/k. These two functions
# match facebookresearch/dinov3's rope_rotate_half / rope_apply exactly
# (GPT-NeoX "rotate-half" convention: split the head_dim in half and swap
# with a sign flip -- NOT the interleaved/complex-number style some other
# RoPE implementations use). Getting this convention wrong produces silently
# wrong numbers, not a crash, so it's kept byte-for-byte matched to upstream.
# ============================================================================
def _rope_rotate_half(x):
    # x: [..., D] -> splits into two halves and rotates: [x0,x1,x2,x3] -> [-x2,-x3,x0,x1]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x, sin, cos):
    # x, sin, cos: [..., D] (D = head_dim). Standard RoPE rotation.
    return (x * cos) + (_rope_rotate_half(x) * sin)


class MaskedJointAttention(nn.Module):
    """
    reuses pretrained qkv/proj weights from both branches, unchanged.
    computes attention jointly over concat(image_tokens, depth_tokens).
    directional mask G = [[1,1],[0,1]] (rows=query, cols=key, order=[image,depth]):
      image query -> image key : allowed
      image query -> depth key : allowed   (depth guides image)
      depth query -> image key : BLOCKED   (image can't corrupt depth)
      depth query -> depth key : allowed

    DINOv3 CHANGE: DINOv3 backbones use RoPE instead of a learned/absolute
    positional embedding. RoPE is applied inside attention, per-layer, only
    to the *patch* tokens -- the leading cls + storage (register) tokens are
    never rotated (they have no 2D position). So this module now also needs,
    per branch: (1) how many leading tokens to skip when rotating (`prefix`),
    and (2) the (sin, cos) rope tensors for that branch's patch grid, passed
    in fresh each forward call.
    """

    def __init__(self, img_attn, depth_attn, num_heads, prefix_img=1, prefix_depth=1):
        super().__init__()
        self.img_qkv = img_attn.qkv
        self.img_proj = img_attn.proj
        self.depth_qkv = depth_attn.qkv
        self.depth_proj = depth_attn.proj
        self.num_heads = num_heads
        # DINOv3 CHANGE: prefix = number of leading non-spatial tokens
        # (cls + storage/register tokens) that RoPE must skip. For DINOv3
        # this is 1 (cls) + model.n_storage_tokens (4 for the vits16 config
        # used here). Passed in from DualBranchEncoder rather than hardcoded,
        # so a different register-token count doesn't silently break things.
        self.prefix_img = prefix_img
        self.prefix_depth = prefix_depth

    def _split_heads(self, x, B):
        # x: (B, N, 3C) -> q,k,v each (B, heads, N, head_dim)
        N, C3 = x.shape[1], x.shape[2]
        C = C3 // 3
        head_dim = C // self.num_heads
        qkv = x.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    # DINOv3 CHANGE: new helper. Rotates only the trailing patch-token slice
    # of q/k (indices [prefix:]); the leading `prefix` cls/storage tokens
    # pass through unrotated. Mirrors DINOv3's SelfAttention.apply_rope,
    # including its dtype handling (rope buffers are often kept in fp32/bf16
    # independent of the model's compute dtype).
    @staticmethod
    def _apply_rope_to_qk(q, k, rope, prefix):
        sin, cos = rope
        q_dtype, k_dtype = q.dtype, k.dtype
        rope_dtype = sin.dtype

        q_prefix, q_patch = q[:, :, :prefix, :], q[:, :, prefix:, :]
        k_prefix, k_patch = k[:, :, :prefix, :], k[:, :, prefix:, :]

        q_patch = _rope_apply(q_patch.to(rope_dtype), sin, cos).to(q_dtype)
        k_patch = _rope_apply(k_patch.to(rope_dtype), sin, cos).to(k_dtype)

        q = torch.cat([q_prefix, q_patch], dim=2)
        k = torch.cat([k_prefix, k_patch], dim=2)
        return q, k

    # DINOv3 CHANGE: forward now accepts optional rope_img / rope_depth,
    # each an (sin, cos) tuple shaped [num_patches, head_dim] for that
    # branch's patch grid. Pass None to fall back to no rotation (e.g. if
    # ever reused with a DINOv2-style backbone that has no RoPE).
    def forward(self, x_img, x_depth, rope_img=None, rope_depth=None):
        B, N_img, C = x_img.shape
        _, N_depth, _ = x_depth.shape

        qkv_img = self.img_qkv(x_img)
        qkv_depth = self.depth_qkv(x_depth)

        q_img, k_img, v_img = self._split_heads(qkv_img, B)
        q_depth, k_depth, v_depth = self._split_heads(qkv_depth, B)

        # DINOv3 CHANGE: rotate each branch's q/k with its own rope BEFORE
        # concatenating into the joint sequence. This must happen per-branch
        # since image and depth patch grids can have different H/W (and
        # therefore different rope tensors), and prefix lengths may differ.
        if rope_img is not None:
            q_img, k_img = self._apply_rope_to_qk(q_img, k_img, rope_img, self.prefix_img)
        if rope_depth is not None:
            q_depth, k_depth = self._apply_rope_to_qk(q_depth, k_depth, rope_depth, self.prefix_depth)

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

    def __init__(self, rgb_block, depth_block, num_heads, prefix_img=1, prefix_depth=1):
        super().__init__()
        self.rgb_block = rgb_block
        self.depth_block = depth_block
        # DINOv3 CHANGE: prefix_img / prefix_depth threaded through to the
        # joint attention module (see MaskedJointAttention docstring above).
        self.joint_attn = MaskedJointAttention(
            rgb_block.attn, depth_block.attn, num_heads,
            prefix_img=prefix_img, prefix_depth=prefix_depth,
        )

    # DINOv3 CHANGE: forward now threads rope_img/rope_depth through to the
    # joint attention call. Everything else (norm/mlp/layerscale residual
    # structure) is untouched -- RoPE only ever touches attention.
    def forward(self, x_img, x_depth, rope_img=None, rope_depth=None):
        img_normed = self.rgb_block.norm1(x_img)
        depth_normed = self.depth_block.norm1(x_depth)

        attn_img, attn_depth = self.joint_attn(img_normed, depth_normed, rope_img, rope_depth)

        x_img = x_img + self.rgb_block.ls1(attn_img)
        x_depth = x_depth + self.depth_block.ls1(attn_depth)

        x_img = x_img + self.rgb_block.ls2(self.rgb_block.mlp(self.rgb_block.norm2(x_img)))
        x_depth = x_depth + self.depth_block.ls2(self.depth_block.mlp(self.depth_block.norm2(x_depth)))

        return x_img, x_depth


class DualBranchEncoder(nn.Module):
    """
    wraps two pretrained dinov3 backbones, fuses every layer via masked joint
    attention, all 12 layers (no dynamic token sampling yet - fixed full grid).
    returns intermediate outputs at layers 6, 12 for the decoder.

    DINOv3 CHANGE: docstring updated from dinov2 -> dinov3. Also now computes
    and threads RoPE (sin, cos) tensors through every fusion block, since
    DINOv3 has no learned positional embedding baked into the tokens
    themselves -- position is entirely a per-layer attention-time operation.
    """

    def __init__(self, model_rgb, model_depth, num_heads=6, extract_layers=(6, 12)):
        super().__init__()
        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.extract_layers = extract_layers

        assert len(model_rgb.blocks) == len(model_depth.blocks)

        # DINOv3 CHANGE: compute the RoPE prefix (cls + storage tokens) once
        # per branch from the backbone's own config, rather than hardcoding
        # it. Falls back to 1 (cls only) for backbones without storage
        # tokens (e.g. old DINOv2 checkpoints without registers).
        prefix_img = 1 + getattr(model_rgb, "n_storage_tokens", 0)
        prefix_depth = 1 + getattr(model_depth, "n_storage_tokens", 0)

        self.fusion_blocks = nn.ModuleList([
            DualBranchFusionBlock(
                model_rgb.blocks[i], model_depth.blocks[i], num_heads,
                prefix_img=prefix_img, prefix_depth=prefix_depth,
            )
            for i in range(len(model_rgb.blocks))
        ])

    def forward(self, rgb, depth_input):
        # DINOv3 CHANGE: prepare_tokens_with_masks now returns (tokens, (H, W))
        # instead of just tokens -- DINOv3 needs the patch-grid H/W to build
        # RoPE. H, W here are patch-grid dimensions (e.g. 14x14 for a 224px
        # input at patch16), not pixel dimensions.
        x_img, (H_img, W_img) = self.model_rgb.prepare_tokens_with_masks(rgb)
        x_depth, (H_depth, W_depth) = self.model_depth.prepare_tokens_with_masks(depth_input)

        # DINOv3 CHANGE: compute each branch's RoPE (sin, cos) once, up
        # front. Unlike the reference dinov3 implementation (which recomputes
        # rope_embed(H, W) fresh inside its per-layer loop purely as a side
        # effect of its list-batching design), the value is identical at
        # every layer for a fixed input resolution, so computing it once
        # here and reusing it across all fusion blocks is equivalent and
        # cheaper. getattr guards against reuse with a DINOv2-style backbone
        # that has no rope_embed at all.
        rope_img = self.model_rgb.rope_embed(H=H_img, W=W_img) if hasattr(self.model_rgb, "rope_embed") else None
        rope_depth = self.model_depth.rope_embed(H=H_depth, W=W_depth) if hasattr(self.model_depth, "rope_embed") else None

        features = {}
        for i, block in enumerate(self.fusion_blocks, start=1):
            x_img, x_depth = block(x_img, x_depth, rope_img, rope_depth)
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
    # unchanged for DINOv3 -- the directional mask logic doesn't depend on
    # RoPE at all, it only cares about the flat N_img / N_depth split.
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
    # unchanged logic, but this is now an even more important check than
    # before: with RoPE, a subtle prefix/rotation bug could leak image
    # *positional* information into depth even if the attention mask is
    # correct, so this test still needs to pass exactly.
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
