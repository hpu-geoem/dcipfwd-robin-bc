"""Schematic of the discretization: Gauss-Lobatto nodes of an order-3 quadrilateral, its low-order-refined submesh,
and a quadtree patch with hanging nodes (2:1 balance)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

GLL3 = np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), 1.0])


def square(ax, x0, y0, size, **kwargs):
    ax.plot([x0, x0 + size, x0 + size, x0, x0], [y0, y0, y0 + size, y0 + size, y0], color=style.INK, **kwargs)


def main() -> None:
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(0.8 * style.DOUBLE, 0.29 * style.DOUBLE), gridspec_kw={"wspace": 0.12})
    nodes = (GLL3 + 1.0) / 2.0
    xx, yy = np.meshgrid(nodes, nodes)
    # (a) High-order element with its nodes.
    square(axes[0], 0, 0, 1, linewidth=1.0)
    axes[0].plot(xx, yy, "o", color=style.BLUE, markersize=4)
    # (b) LOR submesh through the same nodes.
    square(axes[1], 0, 0, 1, linewidth=1.0)
    for v in nodes[1:-1]:
        axes[1].plot([v, v], [0, 1], color=style.INK, linewidth=0.5)
        axes[1].plot([0, 1], [v, v], color=style.INK, linewidth=0.5)
    axes[1].plot(xx, yy, "o", color=style.BLUE, markersize=4)
    # (c) Quadtree with 2:1 balance; hanging vertices are open circles.
    square(axes[2], 0, 0, 1, linewidth=1.0)
    for x0, y0, s in [(0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.25), (0.75, 0.5, 0.25),
                      (0.5, 0.75, 0.25), (0.75, 0.75, 0.25), (0.75, 0.75, 0.125), (0.875, 0.75, 0.125),
                      (0.75, 0.875, 0.125), (0.875, 0.875, 0.125)]:
        square(axes[2], x0, y0, s, linewidth=0.6)
    hanging = [(0.5, 0.75), (0.75, 0.5), (0.875, 0.75), (0.75, 0.875)]
    regular = [(0.5, 0.5), (0.75, 0.75), (0.875, 0.875)]
    for x, y in hanging:
        axes[2].plot(x, y, "o", markerfacecolor="white", markeredgecolor=style.RED, markersize=4)
    for x, y in regular:
        axes[2].plot(x, y, "o", color=style.BLUE, markersize=4)
    titles = ["High-order element", "LOR submesh", "Quadtree, 2:1 balance"]
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")
        ax.set_title(title)
    style.label_panels(axes, x=0.0, y=1.0)
    style.save(fig, "fig-schematic")


if __name__ == "__main__":
    main()
