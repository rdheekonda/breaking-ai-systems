"""Matplotlib visualisation helpers for adversarial attack results."""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.hop_skip_jump import HSJResult
from core.models import prepare_reference
from core.pgd import PGDResult


def _perturbation_map(original: np.ndarray, adversarial: np.ndarray, gain: int = 10) -> np.ndarray:
    """Compute an amplified perturbation difference image."""
    return (np.abs(original.astype(np.float32) - adversarial.astype(np.float32)) * gain).clip(0, 255).astype(np.uint8)


def plot_pgd_result(
    image_path: str,
    pgd: PGDResult,
    save_dir: str | None = None,
) -> None:
    """Plot original, PGD adversarial, and 10x perturbation side-by-side.

    Args:
        image_path: Path to the reference image.
        pgd:        PGDResult from run_pgd().
        save_dir:   If provided, saves adversarial_pgd.png and perturbation_pgd.png.
    """
    ref_256_np, _, _ = prepare_reference(image_path)
    adv_np = np.array(pgd.adversarial_image)
    diff = _perturbation_map(ref_256_np, adv_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(ref_256_np)
    axes[0].set_title("Original (Wolf)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(adv_np)
    axes[1].set_title(f"Adversarial (Granny Smith, p={pgd.confidence:.2f})", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(diff)
    axes[2].set_title(f"Perturbation 10x (L2={pgd.l2_normalized:.1f}, pixel L2={pgd.l2_pixel:.0f})", fontsize=14)
    axes[2].axis("off")

    plt.suptitle("PGD Adversarial Evasion: Wolf -> Granny Smith", fontsize=16)
    plt.tight_layout()
    plt.show()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pgd.adversarial_image.save(os.path.join(save_dir, "adversarial_pgd.png"))
        Image.fromarray(diff).save(os.path.join(save_dir, "perturbation_pgd.png"))
        print(f"Saved to {save_dir}/")


def plot_comparison(
    image_path: str,
    pgd: PGDResult,
    hsj: HSJResult,
) -> None:
    """Plot a 2x3 grid comparing PGD and HSJ adversarial results.

    Args:
        image_path: Path to the reference image.
        pgd:        PGDResult from run_pgd().
        hsj:        HSJResult from run_hsj().
    """
    ref_256_np, _, _ = prepare_reference(image_path)
    pgd_np = np.array(pgd.adversarial_image)
    hsj_np = np.array(hsj.adversarial_image)
    pgd_diff = _perturbation_map(ref_256_np, pgd_np)
    hsj_diff = _perturbation_map(ref_256_np, hsj_np)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top row: PGD
    axes[0, 0].imshow(ref_256_np)
    axes[0, 0].set_title("Original (Wolf)", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pgd_np)
    axes[0, 1].set_title(f"PGD Adversarial (p={pgd.confidence:.2f})", fontsize=13)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pgd_diff)
    axes[0, 2].set_title(f"PGD Perturbation 10x (L2={pgd.l2_normalized:.1f})", fontsize=13)
    axes[0, 2].axis("off")

    # Bottom row: HSJ
    axes[1, 0].imshow(ref_256_np)
    axes[1, 0].set_title("Original (Wolf)", fontsize=13)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hsj_np)
    axes[1, 1].set_title(f"HSJ Adversarial (p={hsj.confidence:.2f})", fontsize=13)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(hsj_diff)
    axes[1, 2].set_title(f"HSJ Perturbation 10x (L2={hsj.l2_normalized:.1f})", fontsize=13)
    axes[1, 2].axis("off")

    plt.suptitle("PGD (White-Box) vs HopSkipJump (Black-Box)", fontsize=16)
    plt.tight_layout()
    plt.show()
