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
from core.tap import (
    TAPResult,
    make_tap_progress_callback,
    print_search_path,
    print_tap_result,
    run_tap,
    send_direct_request,
)
from core.multimodal import print_multimodal_result, send_multimodal_request
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
    "make_tap_progress_callback",
    "print_search_path",
    "print_tap_result",
    "send_direct_request",
    # multimodal
    "print_multimodal_result",
    "send_multimodal_request",
    # transforms
    "add_text_overlay",
    "image_to_base64",
]
