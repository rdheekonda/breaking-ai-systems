"""White-box PGD (Projected Gradient Descent) adversarial attack.

Based on: Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks", arXiv:1706.06083, 2018.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
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
class PGDResult:
    """Outcome of a PGD attack run."""

    success: bool
    adversarial_image: Image.Image
    target_class: int
    target_label: str
    achieved_class: int
    achieved_label: str
    confidence: float
    steps_to_converge: int
    total_steps: int
    l2_normalized: float
    l2_pixel: float
    linf_pixel: float
    execution_time: float


# ── Attack ───────────────────────────────────────────────────────────────────


def run_pgd(
    image_path: str,
    target_class: int = 948,
    target_label: str = "Granny Smith",
    epsilon: float = 8 / 255,
    step_size: float = 1 / 255,
    num_steps: int = 200,
    on_progress: Callable | None = None,
) -> PGDResult:
    """Run targeted PGD on a 256x256 image, attacking only the center crop.

    The perturbation is quantised to uint8 each step so it survives PNG
    encoding.  The image is submitted at 256x256 so the server's
    Resize(256) is a no-op.

    Args:
        image_path:   Path to the reference image.
        target_class: ImageNet class index to target.
        target_label: Human-readable label for the target class.
        epsilon:      L-inf perturbation budget (in [0, 1] scale).
        step_size:    Per-step L-inf step size.
        num_steps:    Maximum PGD iterations.
        on_progress:  Optional callback(step, top_class, confidence, l2).

    Returns:
        PGDResult with the adversarial image and all metrics.
    """
    model = load_model()
    ref_256_np, center_tensor, ref_preprocessed = prepare_reference(image_path)

    target_tensor = torch.tensor([target_class])
    x_adv = center_tensor.clone().detach()
    x_orig = center_tensor.clone().detach()

    t0 = time.time()
    converged_step = num_steps

    for step in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)

        x_norm = (x_adv - MEAN_TENSOR) / STD_TENSOR
        loss = F.cross_entropy(model(x_norm), target_tensor)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv - step_size * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + perturbation, 0, 1)
            x_adv = torch.round(x_adv * 255) / 255

        if on_progress or step == num_steps - 1:
            with torch.no_grad():
                x_check = (x_adv - MEAN_TENSOR) / STD_TENSOR
                probs = F.softmax(model(x_check), dim=1)
                top_class = probs[0].argmax().item()
                conf = probs[0, target_class].item()
                l2 = torch.dist(x_check, ref_preprocessed).item()

            if on_progress:
                on_progress(step, top_class, conf, l2)

            if top_class == target_class:
                converged_step = step + 1
                break

    # ── Final evaluation ─────────────────────────────────────────────────
    with torch.no_grad():
        x_norm = (x_adv - MEAN_TENSOR) / STD_TENSOR
        l2_final = torch.dist(x_norm, ref_preprocessed).item()
        probs = F.softmax(model(x_norm), dim=1)
        confidence = probs[0, target_class].item()
        achieved_class = probs[0].argmax().item()

    adv_center_np = (x_adv[0].detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    adversarial_img = build_256_image(ref_256_np, adv_center_np)

    adv_256_np = np.array(adversarial_img)
    pixel_l2 = float(np.sqrt(np.sum((ref_256_np.astype(float) - adv_256_np.astype(float)) ** 2)))
    pixel_linf = float(np.max(np.abs(ref_256_np.astype(float) - adv_256_np.astype(float))))

    achieved_label = target_label if achieved_class == target_class else f"class {achieved_class}"

    return PGDResult(
        success=achieved_class == target_class,
        adversarial_image=adversarial_img,
        target_class=target_class,
        target_label=target_label,
        achieved_class=achieved_class,
        achieved_label=achieved_label,
        confidence=confidence,
        steps_to_converge=converged_step,
        total_steps=num_steps,
        l2_normalized=l2_final,
        l2_pixel=pixel_l2,
        linf_pixel=pixel_linf,
        execution_time=time.time() - t0,
    )


# ── Rich output ──────────────────────────────────────────────────────────────


def print_pgd_result(result: PGDResult, l2_threshold: float = 150.0) -> None:
    """Render a Rich table summarising PGD attack results."""
    console = Console()

    table = Table(
        title="PGD Attack Results",
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
        "Steps to Converge",
        f"{result.steps_to_converge} / {result.total_steps}",
        "PGD iterations until target was top-1",
    )
    table.add_row(
        "Model Forward Passes",
        str(result.steps_to_converge),
        "Total gradient computations used",
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
    table.add_row("Saved To", "data/adversarial_pgd.png", "Output adversarial image path")

    console.print()
    console.print(table)
