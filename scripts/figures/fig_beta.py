"""Figure: beta = -R dV/dR / V at the surface of a two-layer earth, which the homogeneous condition takes as 1.

Data: figures/data/beta-two-layer.csv from scripts/beta_analysis.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

DATA = style.ROOT / "figures" / "data" / "beta-two-layer.csv"


def main() -> None:
    style.apply()
    data = np.genfromtxt(DATA, delimiter=",", names=True)
    r = data["R_over_h"]
    curves = [("beta_ratio_1000", "1000", style.BLUE, "-"), ("beta_ratio_100", "100", style.GREEN, "--"),
              ("beta_ratio_10", "10", style.ORANGE, "-."), ("beta_ratio_01", "0.1", style.RED, ":"),
              ("beta_ratio_001", "0.01", style.PURPLE, (0, (5, 1, 1, 1, 1, 1)))]
    fig, ax = plt.subplots(figsize=(style.SINGLE, 0.7 * style.SINGLE))
    ax.axhline(1.0, color=style.INK, linewidth=0.5)
    for column, label, color, linestyle in curves:
        ax.plot(r, data[column], color=color, linestyle=linestyle, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_xlabel(r"$R/h$")
    ax.set_ylabel(r"$\beta$")
    ax.set_yticks([0.2, 0.5, 1, 2])
    ax.set_yticklabels(["0.2", "0.5", "1", "2"])
    ax.legend(title=r"$\rho_2/\rho_1$", loc="lower right", ncol=1, handlelength=2.4)
    style.save(fig, "fig-beta")


if __name__ == "__main__":
    main()
