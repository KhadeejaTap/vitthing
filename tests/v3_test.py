import inspect, torch
print("=== DINOv3 Exploration Test ===")
print("Loading DINOv3 ViT-S/16 model...")
try:
    model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', weights='../dinov3-weights/dinov3_vits16.pth')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model with weights: {e}")
    print("Trying without weights...")
    model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
    print("✓ Model loaded successfully (random init)")

print("\n=== STEP 1: Load model + inspect rope module ===")
print("Rope embed forward source:")
print(inspect.getsource(model.rope_embed.forward))
print("Rope embed forward signature:")
print(inspect.signature(model.rope_embed.forward))
print()

print("=== STEP 2: Dump one attn block's forward ===")
# Check the attention block structure
print(f"Attention block type: {type(model.blocks[0].attn)}")
print(f"Attention block class: {model.blocks[0].attn.__class__.__name__}")
print("Available methods:", [m for m in dir(model.blocks[0].attn) if not m.startswith('_')])
print()

# Try to get the forward method source
try:
    attn_forward_source = inspect.getsource(model.blocks[0].attn.forward)
    print("Attention forward source:")
    print(attn_forward_source)
except Exception as e:
    print(f"Could not get forward source: {e}")
    # Try to see what methods are available
    if hasattr(model.blocks[0].attn, 'compute_attention'):
        try:
            compute_source = inspect.getsource(model.blocks[0].attn.compute_attention)
            print("Compute attention source:")
            print(compute_source)
        except Exception as e2:
            print(f"Could not get compute_attention source: {e2}")
print()

print("=== STEP 3: Check token layout ===")
from PIL import Image
import torchvision.transforms as T

# Use a local example image
img_path = "../../examples/frame_0000_rgb.png"
try:
    img = Image.open(img_path).convert("RGB")
    print(f"✓ Loaded image: {img_path}")
except Exception as e:
    print(f"✗ Could not load {img_path}: {e}")
    # Create a dummy image for testing
    import numpy as np
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    print("✓ Using random dummy image for testing")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
x = transform(img).unsqueeze(0)  # add batch dim -> [1,3,224,224]
print(f"Input tensor shape: {x.shape}")

out = model.prepare_tokens_with_masks(x)
print(f"Type of output: {type(out)}, Length: {len(out)}")
for i, o in enumerate(out):
    print(f"  [{i}]: {type(o)}, shape: {getattr(o, 'shape', o)}")

tokens, hw = out
print(f"\nTokens shape: {tokens.shape}")  # [B, N, C]
print(f"HW (patch grid): {hw}")  # (H, W) in patches

# Check token layout: should be [CLS, REGISTER_TOKENS, PATCH_TOKENS]
print(f"\nToken breakdown:")
print(f"  Total tokens (N): {tokens.shape[1]}")
print(f"  Expected: 1 CLS + 4 REGISTER + {hw[0]*hw[1]} PATCH = {1 + 4 + hw[0]*hw[1]}")
print(f"  Actual patch tokens: {hw[0]*hw[1]}")
print(f"  CLS + REGISTER tokens: {tokens.shape[1] - hw[0]*hw[1]}")
print()

print("=== STEP 4: Find where cls/reg get sliced out before rope ===")
# Based on our inspection of fusionv3.py and the rope_embed module,
# we know that DINOv3 applies rope to patch tokens only, skipping prefix tokens (CLS + registers)
# From the token layout: we have 1 CLS + 4 REGISTER + 196 PATCH = 201 tokens
# So prefix length = 1 + 4 = 5

prefix_len = 1 + getattr(model, 'n_storage_tokens', 0)  # CLS + register tokens
print(f"Prefix length (CLS + registers): {prefix_len}")
print(f"Patch tokens start at index: {prefix_len}")
print(f"Number of patch tokens: {tokens.shape[1] - prefix_len}")
print(f"Expected patch tokens from HW: {hw[0]*hw[1]}")
print(f"Match: {tokens.shape[1] - prefix_len == hw[0]*hw[1]}")
print()

print("=== STEP 5: Copy exact slicing + rope call ===")
# Based on fusionv3.py, we can see how rope should be applied:
# 1. Split q/k into prefix and patch tokens
# 2. Apply rope only to patch tokens
# 3. Concatenate back together

print("From fusionv3.py analysis:")
print("- RoPE is applied per-branch to q/k tensors")
print("- Only patch tokens (after prefix) get rotated")
print("- Prefix tokens (CLS + registers) pass through unchanged")
print("- The rope tensors (sin, cos) are computed per branch from patch grid dimensions")
print()

print("=== STEP 6: Sanity check output ===")
print("Testing token preparation and rope application...")

# Get the rope embeddings for our patch grid
H, W = hw
print(f"Patch grid: {H}x{W}")

# Get rope sin/cos for this grid
rope_result = model.rope_embed(H=H, W=W)
print(f"Rope embed output type: {type(rope_result)}")
if isinstance(rope_result, tuple):
    sin, cos = rope_result
    print(f"Sin shape: {sin.shape}, Cos shape: {cos.shape}")
else:
    print(f"Rope result: {rope_result}")

# Test preparing tokens and see what we get
tokens, hw = model.prepare_tokens_with_masks(x)
print(f"Prepared tokens shape: {tokens.shape}")

# Verify token slicing works as expected
prefix_len = 1 + getattr(model, 'n_storage_tokens', 0)  # CLS + register tokens
print(f"Prefix length (CLS + registers): {prefix_len}")
print(f"Patch tokens start at index: {prefix_len}")
print(f"Number of patch tokens: {tokens.shape[1] - prefix_len}")
print(f"Expected patch tokens from HW: {H*W}")
print(f"Match: {tokens.shape[1] - prefix_len == H*W}")

# Test that we can actually extract and work with the tokens
print(f"\nToken analysis:")
print(f"  CLS token shape: {tokens[:, :1, :].shape}")
print(f"  Register tokens shape: {tokens[:, 1:1+getattr(model, 'n_storage_tokens', 0), :].shape}")
print(f"  Patch tokens shape: {tokens[:, 1+getattr(model, 'n_storage_tokens', 0):, :].shape}")

print("\n=== DINOv3 Exploration Completed Successfully ===")
print("Summary:")
print(f"- Model: DINOv3 ViT-S/16")
print(f"- Patch size: 16")
print(f"- Register tokens: {getattr(model, 'n_storage_tokens', 0)}")
print(f"- Uses RoPE: {hasattr(model, 'rope_embed')}")
print(f"- Token layout: 1 CLS + {getattr(model, 'n_storage_tokens', 0)} REG + {H*W} PATCH = {tokens.shape[1]} tokens")
