"""Black-box HopSkipJump adversarial attack.

Based on: Chen et al., "HopSkipJumpAttack: A Query-Efficient Decision-Based
Attack", arXiv:1904.02144, 2020.

ART (Adversarial Robustness Toolbox) is imported lazily inside run_hsj()
so the module can be imported without the heavy dependency at top level.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from rich.table import Table

from core.models import (
    MEAN_TENSOR,
    STD_TENSOR,
    build_256_image,
    load_model,
    prepare_reference,
)


@dataclass
class HSJResult:
    """Outcome of a HopSkipJump attack run."""

    success: bool
    adversarial_image: Image.Image
    target_class: int
    target_label: str
    achieved_class: int
    achieved_label: str
    confidence: float
    total_queries: int
    l2_normalized: float
    l2_pixel: float
    linf_pixel: float
    execution_time: float


# ── Private model wrapper ────────────────────────────────────────────────────


class _QueryCountingModel(nn.Module):
    """Wraps MobileNetV2 to count forward passes and apply normalisation."""

    def __init__(self, base_model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("_std", std.view(1, 3, 1, 1))
        self.query_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.query_count += x.shape[0]
        x = x.float()  # ART passes float64 — conv layers require float32
        return self.base_model((x - self._mean) / self._std)


# ── Attack ───────────────────────────────────────────────────────────────────


def run_hsj(
    image_path: str,
    initial_adversarial: Image.Image,
    target_class: int = 948,
    target_label: str = "Granny Smith",
    max_iter: int = 10,
    max_eval: int = 1000,
    init_eval: int = 100,
    on_progress: Callable | None = None,
) -> HSJResult:
    """Run targeted HopSkipJump starting from an existing adversarial.

    Args:
        image_path:          Path to the reference image.
        initial_adversarial: A PIL image already classified as the target
                             (e.g. the PGD result).  Used as HSJ's starting
                             point to skip random initialisation.
        target_class:        ImageNet class index to target.
        target_label:        Human-readable label for the target class.
        max_iter:            HopSkipJump iterations (boundary walks).
        max_eval:            Max model evaluations per iteration.
        init_eval:           Initial evaluations for gradient estimation.
        on_progress:         Optional callback(queries, achieved_class, confidence).

    Returns:
        HSJResult with the adversarial image and all metrics.
    """
    # Deferred imports — ART is heavy
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import PyTorchClassifier

    model = load_model()
    ref_256_np, center_tensor, ref_preprocessed = prepare_reference(image_path)

    # Wrap model for query counting + built-in normalisation
    wrapped = _QueryCountingModel(model, MEAN_TENSOR.squeeze(), STD_TENSOR.squeeze())
    wrapped.eval()

    classifier = PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )

    # Extract center crop from the initial adversarial as the starting point
    init_np = np.array(initial_adversarial.resize((256, 256), Image.BILINEAR)).astype(np.float32)
    init_center = init_np[16:240, 16:240].copy()
    init_adv = torch.from_numpy(init_center).permute(2, 0, 1).unsqueeze(0).numpy() / 255.0
    init_adv = init_adv.astype(np.float32)

    x_input = center_tensor.numpy().astype(np.float32)
    target_onehot = np.zeros((1, 1000), dtype=np.float32)
    target_onehot[0, target_class] = 1.0

    wrapped.query_count = 0
    t0 = time.time()

    hsj = HopSkipJump(
        classifier=classifier,
        targeted=True,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=True,
    )

    hsj_raw = hsj.generate(x=x_input, y=target_onehot, x_adv_init=init_adv)
    elapsed = time.time() - t0
    total_queries = wrapped.query_count

    # ── Evaluate ─────────────────────────────────────────────────────────
    hsj_tensor = torch.from_numpy(hsj_raw).float()
    with torch.no_grad():
        hsj_norm = (hsj_tensor - MEAN_TENSOR) / STD_TENSOR
        probs = F.softmax(model(hsj_norm), dim=1)
        confidence = probs[0, target_class].item()
        achieved_class = probs[0].argmax().item()
        l2_norm = torch.dist(hsj_norm, ref_preprocessed).item()

    l2_pixel = float(np.sqrt(np.sum(((x_input - hsj_raw) * 255) ** 2)))
    linf_pixel = float(np.max(np.abs(x_input - hsj_raw)) * 255)

    adv_center_np = (hsj_raw[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    adversarial_img = build_256_image(ref_256_np, adv_center_np)

    achieved_label = target_label if achieved_class == target_class else f"class {achieved_class}"

    if on_progress:
        on_progress(total_queries, achieved_class, confidence)

    return HSJResult(
        success=achieved_class == target_class,
        adversarial_image=adversarial_img,
        target_class=target_class,
        target_label=target_label,
        achieved_class=achieved_class,
        achieved_label=achieved_label,
        confidence=confidence,
        total_queries=total_queries,
        l2_normalized=l2_norm,
        l2_pixel=l2_pixel,
        linf_pixel=linf_pixel,
        execution_time=elapsed,
    )


# ── Rich output ──────────────────────────────────────────────────────────────


def print_hsj_result(result: HSJResult, l2_threshold: float = 150.0) -> None:
    """Render a Rich table summarising HopSkipJump attack results."""
    console = Console()

    table = Table(
        title="HopSkipJump Attack Results",
        show_header=True,
        header_style="bold white on blue",
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", width=26)
    table.add_column("Value", style="cyan", width=28)
    table.add_column("Description", width=36)

    status = "[bold green]++ SUCCESS[/]" if result.success else "[bold red]-- FAILED[/]"
    table.add_row("Status", status, "Did the attack achieve the target class?")
    table.add_row("Original Class", "timber wolf (269)", "Model's prediction on clean input")
    table.add_row(
        "Adversarial Class",
        f"[bold magenta]{result.achieved_label} ({result.achieved_class})[/]",
        "Model's prediction on perturbed input",
    )
    table.add_row(
        "Granny Smith Confidence",
        f"[bold]{result.confidence:.4f}[/]",
        "Softmax probability for target class",
    )
    table.add_row(
        "Total Model Queries",
        str(result.total_queries),
        "Forward passes used (no gradients)",
    )

    l2_style = "green" if result.l2_normalized < l2_threshold else "red"
    table.add_row(
        "L2 (normalized tensor)",
        f"[{l2_style}]{result.l2_normalized:.2f}[/] / {l2_threshold}",
        "Euclidean distance after ImageNet normalization",
    )
    table.add_row("L2 (pixel space)", f"{result.l2_pixel:.2f}", "Euclidean distance in raw pixel values")
    table.add_row("L-inf (pixel)", f"{result.linf_pixel:.1f} / 255", "Max per-pixel change (worst channel)")
    table.add_row("Wall Time", f"{result.execution_time:.1f}s", "Total compute time on CPU")

    console.print()
    console.print(table)
