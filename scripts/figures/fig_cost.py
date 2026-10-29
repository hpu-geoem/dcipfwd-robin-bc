"""Accuracy against cost of the five boundary treatments (fig-cost).

(a) Example 1 (3-D): largest dipole-dipole truncation error against the wall time of the DC and IP solutions;
(b) Example 3: 99th percentile of the rho_a truncation error against wall time, 25 fitted wavenumbers;
(c, d) Example 4: 99th percentiles of the rho_a and m_a errors against wall time;
(e) degrees of freedom against the outer extent, relative to the smallest extent, for the three meshes (DS runs).
Each line joins the extents of one treatment (smallest extent at the left end). Times are the wall times reported by
dcipfwd (refinement, setup, preconditioners and all solves of both passes; maximum over processes); each example ran
on one exclusive node. Data: results/example1-3d, results/example3, results/example4; writes figures/data/cost.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics  # noqa: E402
import style  # noqa: E402

SERIES = {
    "example1": ("results/example1-3d", "{m}-{e}", "results/example1-3d/le-20000", (500, 1000, 2000, 5000, 20000)),
    "example3": ("results/example3", "{m}-layered4-{e}", "results/example3/le-dense-200000", (500, 1000, 2000, 5000, 10000)),
    "example4": ("results/example4", "{m}-{e}", "results/example4/le-200000", (500, 1000, 2000, 5000, 10000)),
}


def points(example: str, method: str):
    directory, pattern, reference, extents = SERIES[example]
    ref, _ = metrics.read(reference)
    rows = []
    metrics.require([f"{directory}/{pattern.format(m=method, e=extent)}" for extent in extents
                     if not (extent == extents[-1] and example == "example1" and method in ("le", "li"))])
    for extent in extents:
        path = f"{directory}/{pattern.format(m=method, e=extent)}"
        if path.replace("-layered4", "").endswith(reference.split("/")[-1]) or (method == "li" and extent == extents[-1]
                                                                                 and example == "example1"):
            continue  # the reference itself, and LI on the reference domain
        table, report = metrics.read(path)
        e = metrics.errors(table, ref, path)
        c = metrics.cost(report)
        rows.append((extent, c["wall"], c["dofs"], e["dipole"]["rho"]["max"], e["all"]["rho"]["p99"], e["all"]["ma"]["p99"]))
    return np.array(rows)


def main() -> None:
    style.apply()
    fig, axes = plt.subplots(1, 5, figsize=(style.DOUBLE, 0.3 * style.DOUBLE),
                             gridspec_kw={"wspace": 0.62, "top": 0.78})
    csv = ["example,method,extent_m,wall_s,dofs,dd_max,rho_p99,ma_p99_mVV"]
    panels = [("example1", 3, "(a) Ex. 1, DD max."), ("example3", 4, r"(b) Ex. 3, $\rho_a$ 99%"),
              ("example4", 4, r"(c) Ex. 4, $\rho_a$ 99%"), ("example4", 5, r"(d) Ex. 4, $m_a$ 99%")]
    for method, kwargs in style.BOUNDARIES.items():
        for ax, (example, column, title) in zip(axes, panels):
            data = points(example, method)
            if len(data):
                ax.plot(data[:, 1], data[:, column], **kwargs)
                ax.plot(data[0, 1], data[0, column], marker=kwargs["marker"], color=kwargs["color"], markersize=5,
                        markerfacecolor="none", linestyle="none")
            ax.set_title(title, loc="left")
        for example in SERIES:
            for row in points(example, method):
                csv.append(f"{example},{method}," + ",".join(f"{v:.6g}" for v in row))
    for ax, (example, column, _) in zip(axes, panels):
        if example == "example1":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Error (mV/V)" if column == 5 else "Relative error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.43, 1.0), fontsize=7)
    for example, (color, marker) in zip(SERIES, ((style.DARK, "o"), (style.BLUE, "s"), (style.RED, "^"))):
        data = points(example, "ds")
        if len(data):
            axes[4].plot(data[:, 0] / 1000.0, data[:, 2] / data[0, 2], color=color, marker=marker,
                         label={"example1": "Ex. 1 (3-D)", "example3": "Ex. 3", "example4": "Ex. 4"}[example])
    axes[4].set_xscale("log")
    axes[4].set_xlabel("Extent (km)")
    axes[4].set_ylabel("DoFs / DoFs(0.5 km)")
    axes[4].set_title("(e) Padding")
    axes[4].legend(loc="upper left", fontsize=6.5)
    axes[4].xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    style.save(fig, "fig-cost")
    (style.ROOT / "figures" / "data" / "cost.csv").write_text("\n".join(csv) + "\n")


if __name__ == "__main__":
    main()
