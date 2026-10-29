"""Figures for example 3 (Overthrust section on a quadtree mesh, 2.5D).

- fig-example3-model: resistivity and chargeability of the section with the quadtree cells and the electrodes;
- fig-example3-convergence: largest truncation error against the extent (DS, DE, LS, LE, LI; dense rule);
- fig-example3-quadrature: median error and background residual against the number of wavenumbers (LE, Gauss and fitted
  weights, 200 km);
- fig-example3-pseudo: reference dipole-dipole pseudosections and the errors of a usual and a compact set-up.
Data: models/e8-overthrust/octree-section (model-500.cells from octree_mesh), results/example3 (scripts/paper_runs.py,
group example3). Also writes figures/data/example3-convergence.csv and figures/data/example3-quadrature.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dcipio  # noqa: E402
import octree_views  # noqa: E402
import pseudo  # noqa: E402
import style  # noqa: E402

MODEL = style.ROOT / "models" / "e8-overthrust" / "octree-section"
RESULTS = style.ROOT / "results" / "example3"
EXTENTS = [500, 1000, 2000, 5000, 10000]
SPACING = 100.0  # electrode spacing (m)
USUAL, PROPOSED, DATA = "ds-log4-5000", "le-layered4-1000", "li-layered4-1000"
REFERENCE = "le-dense-200000"


def electrodes() -> np.ndarray:
    lines = [ln.split("#")[0].strip() for ln in (MODEL / "survey.txt").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    n = int(lines[0])
    return np.array([[float(v) for v in ln.split()] for ln in lines[1:1 + n]])


def read(name: str):
    return dcipio.read_result(RESULTS / name)


def model_figure() -> None:
    cells = octree_views.read_cells(MODEL / "model-500.cells")
    core = cells[cells[:, -1] == 1]
    polygons = np.stack([core[:, [0, 1]], core[:, [2, 1]], core[:, [2, 3]], core[:, [0, 3]]], axis=1) / 1000.0
    x = electrodes()[:, 1] / 1000.0
    fig = plt.figure(figsize=(style.DOUBLE, 0.43 * style.DOUBLE))
    # Two square map axes, each with a slim colorbar on its right.
    width, height, bottom = 0.36, 0.36 * style.DOUBLE / (0.43 * style.DOUBLE), 0.13
    panels = [(core[:, 4], style.RESISTIVITY_CMAP, Normalize(1.1, 3.65), r"$\log_{10}[\rho/(\Omega\,\mathrm{m})]$"),
              (core[:, 5] * 1000.0, style.CHARGEABILITY_CMAP, Normalize(20.0, 150.0), r"$\eta$ (mV/V)")]
    axes = []
    for index, (values, cmap, norm, label) in enumerate(panels):
        left = 0.07 + index * 0.5
        ax = fig.add_axes([left, bottom, width, height])
        bar = fig.add_axes([left + width + 0.012, bottom, 0.013, height])
        mesh = octree_views.draw_slice(ax, polygons, values, cmap, norm)
        ax.plot(x, np.zeros_like(x), "v", color=style.INK, markersize=1.4, clip_on=False, zorder=5)
        ax.set_xlim(7.6, 12.4)
        ax.set_ylim(4.8, 0.0)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("Depth (km)")
        fig.colorbar(mesh, cax=bar, label=label)
        axes.append(ax)
    style.label_panels(axes)
    style.save(fig, "fig-example3-model")


def truncation_errors(variant: str, reference) -> np.ndarray:
    pole = reference.b < 0
    values = []
    for extent in EXTENTS:
        table, _ = read(f"{variant}-dense-{extent}")
        rel = np.abs(table.rho_a / reference.rho_a - 1.0)
        ip = np.abs(table.m_a - reference.m_a) * 1000.0
        values.append((rel[~pole].max(), rel[pole].max(), ip.max()))
    return np.array(values)


def convergence_figure() -> None:
    reference, _ = read(REFERENCE)
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 0.36 * style.DOUBLE),
                             gridspec_kw={"wspace": 0.38, "top": 0.8})
    rows = ["variant,extent_m,dipole_dipole_rel,pole_dipole_rel,ma_abs_mVV"]
    for key in style.BOUNDARIES:
        values = truncation_errors(key, reference)
        for extent, v in zip(EXTENTS, values):
            rows.append(f"{key},{extent},{v[0]:.4e},{v[1]:.4e},{v[2]:.4e}")
        for column in range(3):
            axes[column].plot(np.array(EXTENTS) / 1000.0, values[:, column], **style.BOUNDARIES[key])
    for column, title in enumerate(["Dipole-dipole", "Pole-dipole", r"$m_a$"]):
        ax = axes[column]
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Extent (km)")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
        ax.set_ylabel("Error (mV/V)" if column == 2 else "Relative error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995))
    style.label_panels(axes)
    style.save(fig, "fig-example3-convergence")
    (style.ROOT / "figures" / "data" / "example3-convergence.csv").write_text("\n".join(rows) + "\n")


def quadrature_figure() -> None:
    reference, _ = read(REFERENCE)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 0.75 * style.SINGLE))
    rows = ["rule,wavenumbers,median_rel,background_residual_dc"]
    for rule, color, marker, name in (("log", style.BLUE, "o", "Gauss"), ("layered", style.RED, "^", "DC fit"),
                                      ("layered-joint", style.GREEN, "D", "Joint fit")):
        counts, actual, prior = [], [], []
        for ppd in (2, 3, 4):
            table, report = read(f"le-{rule}{ppd}-200000")
            counts.append(report["passes"][0]["solvers"][0]["wavenumbers"])
            actual.append(np.median(np.abs(table.rho_a / reference.rho_a - 1.0)))
            prior.append(report["rule_dc"]["error"])
            rows.append(f"{rule},{counts[-1]},{actual[-1]:.4e},{prior[-1]:.4e}")
        ax.plot(counts, actual, color=color, marker=marker, label=name)
        ax.plot(counts, prior, color=color, marker=marker, linestyle="--", fillstyle="none", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of wavenumbers $N_k$")
    ax.set_ylabel("Relative error")
    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 1.0, counts[-1] + 1.0)
    rules = ax.legend(loc="upper right")
    ax.add_artist(rules)
    styles = [plt.Line2D([], [], color=style.INK, label="Data, median"),
              plt.Line2D([], [], color=style.INK, linestyle="--", label=r"Background, $\epsilon_{\rm bg}$")]
    ax.legend(handles=styles, loc="lower left")
    style.save(fig, "fig-example3-quadrature")
    (style.ROOT / "figures" / "data" / "example3-quadrature.csv").write_text("\n".join(rows) + "\n")


def pseudo_figure() -> None:
    e = electrodes()
    reference, _ = read(REFERENCE)
    usual, _ = read(USUAL)
    proposed, _ = read(PROPOSED)
    panels = [(reference.rho_a, "rho"), (reference.m_a * 1000.0, "eta"),
              (100.0 * (usual.rho_a / reference.rho_a - 1.0), "err"), ((usual.m_a - reference.m_a) * 1000.0, "err_ma"),
              (100.0 * (proposed.rho_a / reference.rho_a - 1.0), "err"),
              ((proposed.m_a - reference.m_a) * 1000.0, "err_ma")]
    dd = reference.b >= 0
    err_limit = max(np.nanmax(np.abs(p[0][dd])) for p in panels if p[1] == "err")
    ma_limit = max(np.nanmax(np.abs(p[0][dd])) for p in panels if p[1] == "err_ma")
    fig, axes = plt.subplots(3, 2, figsize=(style.DOUBLE, 0.66 * style.DOUBLE),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.62})
    for ax, (values, kind) in zip(axes.ravel(), panels):
        mids, levels, array = pseudo.grid(reference, e, values, "dd")
        depths = pseudo.pseudodepth(levels, SPACING, "dd")
        if kind == "rho":
            array = np.log10(array)
            cmap, norm, label = style.RESISTIVITY_CMAP, None, r"$\log_{10}[\rho_a/(\Omega\,\mathrm{m})]$"
        elif kind == "eta":
            cmap, norm, label = style.CHARGEABILITY_CMAP, None, r"$m_a$ (mV/V)"
        elif kind == "err":
            cmap, norm, label = style.ERROR_CMAP, TwoSlopeNorm(0.0, -err_limit, err_limit), r"$\Delta\rho_a$ (%)"
        else:
            cmap, norm, label = style.ERROR_CMAP, TwoSlopeNorm(0.0, -ma_limit, ma_limit), r"$\Delta m_a$ (mV/V)"
        mesh = pseudo.draw(ax, mids / 1000.0, depths, array, cmap, norm)
        ax.set_ylim(depths[-1] + SPACING / 4.0, 0.0)
        ax.set_xlim(8.0, 12.0)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("Pseudodepth (m)")
        fig.colorbar(mesh, ax=ax, label=label, pad=0.02)
    for row, title in enumerate(("Reference", "DS, 5 km, Gauss", "LE, 1 km, fitted")):
        for ax in axes[row]:
            ax.set_title(title, fontsize=7.5)
    style.label_panels(axes.ravel())
    style.save(fig, "fig-example3-pseudo")


if __name__ == "__main__":
    style.apply()
    model_figure()
    convergence_figure()
    quadrature_figure()
    pseudo_figure()
