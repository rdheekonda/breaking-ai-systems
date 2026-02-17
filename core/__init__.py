"""Core attack implementations for Breaking AI Systems demos."""

from core.display import print_comparison
from core.hop_skip_jump import HSJResult, print_hsj_result, run_hsj
from core.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SERVER_PREPROCESS,
    build_256_image,
    load_model,
    predict_top_k,
    prepare_reference,
)
from core.pgd import PGDResult, print_pgd_result, run_pgd
from core.tap import TAPResult, print_search_path, run_tap
from core.transforms import add_text_overlay, image_to_base64
from core.utils import submit_flag
from core.visual import plot_comparison, plot_pgd_result

__all__ = [
    # models
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SERVER_PREPROCESS",
    "build_256_image",
    "load_model",
    "predict_top_k",
    "prepare_reference",
    # pgd
    "PGDResult",
    "print_pgd_result",
    "run_pgd",
    # hop_skip_jump
    "HSJResult",
    "print_hsj_result",
    "run_hsj",
    # display
    "print_comparison",
    # visual
    "plot_pgd_result",
    "plot_comparison",
    # utils
    "submit_flag",
    # tap
    "run_tap",
    "TAPResult",
    "print_search_path",
    # transforms
    "add_text_overlay",
    "image_to_base64",
]
