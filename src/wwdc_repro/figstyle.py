"""Paper figure style, ported from the historical examples/spectra/symbolic_regression/figstyle.py.

The manuscript's anatomy and CMD images were rendered with scienceplots' ``science``
style in its ``no-latex`` form (the historical ``use_science()`` fell back to it because
``dvipng`` is unavailable), so that form is fixed here rather than probed at run time.
"""

from __future__ import annotations

import scienceplots  # noqa: F401  -- registers the styles with matplotlib

STYLE = ["science", "no-latex"]

INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
SURFACE = "white"


def tidy(ax, grid_axis: str | None = "both", *, labelsize: float = 8) -> None:
    """Light grid over the science style's box, and muted tick labels."""
    if grid_axis:
        ax.grid(
            True, axis=grid_axis, color=INK_MUTED, alpha=0.18, linewidth=0.5, zorder=0
        )
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=labelsize)


# Arm palette, order and labels, as in the historical figstyle.py.
ARM_COLOUR = {
    "cond+z": "#2a78d6",
    "orig": "#eb6834",
    "uncond": "#1baf7a",
    "z": "#8a4fbf",
    "cond": "#c8a415",
    "orig+z": "#9b1b4a",
}
ARM_ORDER = ["cond+z", "orig+z", "orig", "uncond", "z", "cond"]
ARM_LABEL = {
    "cond+z": "conditional + z",
    "orig+z": "original + z",
    "orig": "original",
    "uncond": "unconditional",
    "z": "redshift only",
    "cond": "conditional only",
}


def frontier_legend(ax) -> None:
    leg = ax.legend(
        loc="lower right", ncol=2, frameon=True, fontsize=7.5, handlelength=1.8,
        borderpad=0.5, labelspacing=0.4, columnspacing=1.2,
    )
    frame = leg.get_frame()
    frame.set_edgecolor(INK_MUTED)
    frame.set_linewidth(0.6)
    frame.set_facecolor(SURFACE)
    frame.set_alpha(1.0)
