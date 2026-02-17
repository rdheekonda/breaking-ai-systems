"""Shared model loading and preprocessing for adversarial evasion demos.

Provides a single cached MobileNetV2 instance, ImageNet constants, and
helper functions so notebooks never duplicate model-loading boilerplate.
"""

import functools

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models.mobilenet import MobileNet_V2_Weights

# ── ImageNet constants ───────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mean/std as (1, 3, 1, 1) tensors for direct arithmetic on NCHW batches.
MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD_TENSOR = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

# The exact preprocessing pipeline the Crucible server applies.
SERVER_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model cache ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Return a cached, eval-mode MobileNetV2 with IMAGENET1K_V2 weights."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.eval()
    return model


# ── Image helpers ────────────────────────────────────────────────────────────

def prepare_reference(image_path: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Load a reference image and return everything needed for PGD / HSJ.

    The image is resized to 256x256 (making the server's Resize(256) a no-op)
    and the center 224x224 crop is extracted as a float32 tensor in [0, 1].

    Returns:
        ref_256_np:          (256, 256, 3) uint8 numpy array.
        center_tensor:       (1, 3, 224, 224) float32 tensor in [0, 1].
        preprocessed_tensor: (1, 3, 224, 224) ImageNet-normalised tensor
                             (for L2 distance comparisons).
    """
    img = Image.open(image_path).convert("RGB")
    ref_256 = img.resize((256, 256), Image.BILINEAR)
    ref_256_np = np.array(ref_256).astype(np.float32)

    # Center 224x224 crop as [0, 1] tensor
    center_np = ref_256_np[16:240, 16:240].copy()
    center_tensor = torch.from_numpy(center_np).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Full preprocessing (for L2 baseline)
    preprocessed_tensor = SERVER_PREPROCESS(img).unsqueeze(0)

    return np.array(ref_256).astype(np.uint8), center_tensor, preprocessed_tensor


def predict_top_k(
    image_path: str,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Run local MobileNetV2 inference and return top-k (class_idx, prob) pairs."""
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    x = SERVER_PREPROCESS(img).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
        top = torch.topk(probs, k)

    return [(idx.item(), prob.item()) for idx, prob in zip(top.indices[0], top.values[0])]


def build_256_image(ref_256_np: np.ndarray, adv_center: np.ndarray) -> Image.Image:
    """Paste an adversarial center crop back into a 256x256 reference frame.

    Args:
        ref_256_np:  (256, 256, 3) uint8 numpy — the original 256x256 image.
        adv_center:  (224, 224, 3) uint8 numpy — the adversarial center crop.

    Returns:
        PIL Image (256x256, RGB).
    """
    out = ref_256_np.copy()
    out[16:240, 16:240] = adv_center
    return Image.fromarray(out)
