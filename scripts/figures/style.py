"""Shared figure style: SciencePlots 'science' style with black axes, inward ticks, a Times-like font, the SciencePlots
colour cycle, and the colormaps of all model and pseudosection panels."""

from __future__ import annotations

import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"

# SciencePlots 'science' colour cycle.
BLUE, GREEN, ORANGE, RED, PURPLE, DARK = "#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747"
INK = "black"

# Boundary treatments: homogeneous (D) or layered (L) coefficient; shared reference (S), per-electrode references (E),
# or per-electrode layered boundary data with the shared coefficient (I).
BOUNDARIES = {
    "ds": dict(label="DS", color=BLUE, marker="o", linestyle="--"),
    "de": dict(label="DE", color=PURPLE, marker="v", linestyle=":"),
    "ls": dict(label="LS", color=ORANGE, marker="s", linestyle="-."),
    "le": dict(label="LE", color=RED, marker="^", linestyle="-", zorder=3),
    "li": dict(label="LI", color=GREEN, marker="D", linestyle=(0, (5, 2)), markerfacecolor="none", markersize=4.5),
}

RESISTIVITY_CMAP = "Spectral_r"
CHARGEABILITY_CMAP = "magma_r"
ERROR_CMAP = "RdBu_r"

MM = 1.0 / 25.4
SINGLE = 90 * MM
DOUBLE = 190 * MM


def apply() -> None:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })


def label_panels(axes, x: float = 0.0, y: float = 1.02, start: int = 0) -> None:
    """Write (a), (b), ... above the upper-left corner of each axes."""
    for index, ax in enumerate(axes):
        ax.text(x, y, f"({string.ascii_lowercase[start + index]})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
