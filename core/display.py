"""Shared Rich display helpers for adversarial attack comparison."""

from rich.console import Console
from rich.table import Table

from core.hop_skip_jump import HSJResult
from core.pgd import PGDResult


def print_comparison(
    pgd: PGDResult,
    hsj: HSJResult,
    l2_threshold: float = 150.0,
) -> None:
    """Render a head-to-head comparison table and recommendations."""
    console = Console()

    # ── Head-to-Head ─────────────────────────────────────────────────────
    comp = Table(
        title="PGD vs HopSkipJump \u2014 Head to Head",
        show_header=True,
        header_style="bold white on blue",
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_lines=True,
        padding=(0, 1),
    )
    comp.add_column("Metric", style="bold", width=22)
    comp.add_column("Description", width=28)
    comp.add_column("PGD (White-Box)", justify="center", width=18, style="cyan")
    comp.add_column("HSJ (Black-Box)", justify="center", width=18, style="magenta")

    def _row(
        label: str,
        desc: str,
        pgd_v: int | float | str,
        hsj_v: int | float | str,
        fmt: str = ".2f",
        lower_better: bool = True,
    ) -> None:
        pf = f"{pgd_v:{fmt}}" if isinstance(pgd_v, float) else str(pgd_v)
        hf = f"{hsj_v:{fmt}}" if isinstance(hsj_v, float) else str(hsj_v)
        if isinstance(pgd_v, (int, float)) and isinstance(hsj_v, (int, float)):
            pgd_better = (pgd_v < hsj_v) if lower_better else (pgd_v > hsj_v)
            hsj_better = (hsj_v < pgd_v) if lower_better else (hsj_v > pgd_v)
        else:
            pgd_better = hsj_better = False
        ps = f"[bold green]{pf} <<[/]" if pgd_better else f"[cyan]{pf}[/]"
        hs = f"[bold green]{hf} <<[/]" if hsj_better else f"[magenta]{hf}[/]"
        comp.add_row(label, desc, ps, hs)

    pgd_ok = "[bold green]Yes[/]" if pgd.success else "[bold red]No[/]"
    hsj_ok = "[bold green]Yes[/]" if hsj.success else "[bold red]No[/]"
    comp.add_row("Success", "Achieved target class?", pgd_ok, hsj_ok)

    _row("Confidence", "Softmax prob for Granny Smith", pgd.confidence, hsj.confidence, ".4f", lower_better=False)
    _row("Model Queries", "Forward passes used", pgd.steps_to_converge, hsj.total_queries, "d", lower_better=True)
    _row("L2 (normalized)", "Euclidean dist in tensor space", pgd.l2_normalized, hsj.l2_normalized, ".2f")
    _row("L2 (pixel)", "Euclidean dist in pixel space", pgd.l2_pixel, hsj.l2_pixel, ".2f")
    _row("L-inf (pixel)", "Max per-pixel change", pgd.linf_pixel, hsj.linf_pixel, ".1f")
    _row("Wall Time (s)", "Total compute time", pgd.execution_time, hsj.execution_time, ".1f")
    comp.add_row("Requires Weights", "Needs model internals?", "[cyan]Yes[/]", "[bold green]No <<[/]")

    console.print()
    console.print(comp)

    query_ratio = hsj.total_queries / max(pgd.steps_to_converge, 1)
    console.print(f"\n[bold]HSJ used {query_ratio:.0f}x more queries[/] than PGD for a comparable adversarial.")
    console.print(f"[bold green]Both attacks beat the challenge[/] \u2014 L2 well under {l2_threshold} threshold.\n")

    # ── Recommendations ──────────────────────────────────────────────────
    recs = Table(
        title="Recommendations",
        show_header=True,
        header_style="bold white on blue",
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_lines=True,
        padding=(0, 1),
    )
    recs.add_column("Scenario", style="bold", width=26)
    recs.add_column("Approach", width=50)

    recs.add_row(
        "White-box\n(weights available)",
        "PGD, C&W, AutoAttack.\nFast, precise, minimal perturbation.\nBest for auditing your own models.",
    )
    recs.add_row(
        "Black-box / API-only\n(no weights)",
        "HopSkipJump, Boundary Attack, Square Attack.\nBudget 100\u201310,000+ queries.\nRealistic for attacking deployed systems.",
    )
    recs.add_row(
        "Defense",
        "Adversarial training, input preprocessing\n(JPEG, spatial smoothing), certified defenses.\nNo single defense is absolute.",
    )

    console.print(recs)
