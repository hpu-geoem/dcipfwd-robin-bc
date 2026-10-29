"""Figures for example 4 (Overthrust volume on an octree mesh, 3D).

- fig-example4-model: 3D cutaways of resistivity and chargeability with the octree cells and electrodes, a depth slice
  and the vertical section under the central line;
- fig-example4-responses: plan-view maps of the reference dipole-dipole apparent resistivity and chargeability at two
  pseudodepths, and fences of the nine dipole-dipole pseudosections;
- fig-example4-errors: truncation error against the extent, cumulative error distributions on the 1 km domain, and
  plan-view error maps of DS and LE.
Data: models/e8-overthrust/octree-volume (model-500.cells from octree_mesh), results/example4 (scripts/paper_runs.py,
groups example4-main and example4-aux). Panels are placed explicitly so that axes of the same kind have the same size.
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
import metrics  # noqa: E402
import octree_views  # noqa: E402
import pseudo  # noqa: E402
import style  # noqa: E402

MODEL = style.ROOT / "models" / "e8-overthrust" / "octree-volume"
RESULTS = style.ROOT / "results" / "example4"
RENDER = style.ROOT / "figures" / "data" / "render"
EXTENTS = [500, 1000, 2000, 5000, 10000]
REFERENCE = "le-200000"
PER_LINE = 41
SPACING = 100.0  # electrode spacing (m)
COMPARE = 1000
LEVELS = (2, 6)
RHO_CLIM, ETA_CLIM = (1.1, 3.65), (20.0, 150.0)


def electrodes() -> np.ndarray:
    lines = [ln.split("#")[0].strip() for ln in (MODEL / "survey.txt").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    n = int(lines[0])
    return np.array([[float(v) for v in ln.split()] for ln in lines[1:1 + n]])


def read(name: str):
    return dcipio.read_result(RESULTS / name)


def dd_points(table, e: np.ndarray, level: int):
    """Midpoints (x, y) in km and row mask of the dipole-dipole data with separation n = level."""
    dd = table.b >= 0
    a, b, m, n = (table[k].astype(int) for k in ("a", "b", "m", "n"))
    rows = dd & (m - b == level)
    x = 0.25 * (e[a, 0] + e[b, 0] + e[m, 0] + e[n, 0])
    return x[rows] / 1000.0, e[a[rows], 1] / 1000.0, rows


def depth_label(level: int) -> str:
    return f"Pseudodepth {pseudo.pseudodepth(level, SPACING, 'dd'):.0f} m"


def panel_label(fig, ax, letter: str) -> None:
    box = ax.get_position()
    fig.text(box.x0, box.y1 + 0.006, f"({letter})", ha="left", va="bottom", fontsize=8)


def model_figure() -> None:
    cells = octree_views.read_cells(MODEL / "model-500.cells")
    e = electrodes()
    RENDER.mkdir(parents=True, exist_ok=True)
    octree_views.render_cutaway(cells, 6, style.RESISTIVITY_CMAP, RHO_CLIM, RENDER / "example4-rho.png", e)
    octree_views.render_cutaway(cells, 7, style.CHARGEABILITY_CMAP, ETA_CLIM, RENDER / "example4-eta.png", e)
    width, height = style.DOUBLE, 0.74 * style.DOUBLE
    fig = plt.figure(figsize=(width, height))
    crop = ((170, 1300), (120, 1700))
    rho_norm, eta_norm = Normalize(*RHO_CLIM), Normalize(*ETA_CLIM)
    top = [fig.add_axes([0.0, 0.5, 0.43, 0.47]), fig.add_axes([0.5, 0.5, 0.43, 0.47])]
    for ax, png, norm, cmap, label, x_bar in ((top[0], "example4-rho.png", rho_norm, style.RESISTIVITY_CMAP,
                                               r"$\log_{10}[\rho/(\Omega\,\mathrm{m})]$", 0.435),
                                              (top[1], "example4-eta.png", eta_norm, style.CHARGEABILITY_CMAP,
                                               r"$\eta$ (mV/V)", 0.935)):
        octree_views.image_axes(ax, RENDER / png, crop)
        bar = fig.add_axes([x_bar, 0.57, 0.012, 0.33])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, label=label)
    # Depth slice (square) and section (4.8 km by 3 km) with the same height.
    h = 0.37
    w_slice = h * height / width
    slice_ax = fig.add_axes([0.07, 0.07, w_slice, h])
    section_ax = fig.add_axes([0.07 + w_slice + 0.1, 0.07, w_slice * 4.8 / 3.0, h])
    polygons, hit = octree_views.slice_polygons(cells, 2, 60.0, (0, 1))
    core = hit[:, -1] == 1
    mesh = octree_views.draw_slice(slice_ax, polygons[core], hit[core, 6], style.RESISTIVITY_CMAP, rho_norm)
    slice_ax.plot(e[:, 0] / 1000.0, e[:, 1] / 1000.0, ".", color=style.INK, markersize=0.8)
    slice_ax.set_xlim(7.6, 12.4)
    slice_ax.set_ylim(7.6, 12.4)
    slice_ax.set_xlabel("x (km)")
    slice_ax.set_ylabel("y (km)")
    polygons, hit = octree_views.slice_polygons(cells, 1, 10010.0, (0, 2))
    core = hit[:, -1] == 1
    octree_views.draw_slice(section_ax, polygons[core], hit[core, 6], style.RESISTIVITY_CMAP, rho_norm)
    section_ax.set_xlim(7.6, 12.4)
    section_ax.set_ylim(3.0, 0.0)
    section_ax.set_xlabel("x (km)")
    section_ax.set_ylabel("Depth (km)")
    box = section_ax.get_position()
    fig.colorbar(mesh, cax=fig.add_axes([box.x1 + 0.012, box.y0, 0.012, box.height]), label=r"$\log_{10}[\rho/(\Omega\,\mathrm{m})]$")
    for ax, letter in zip(top + [slice_ax, section_ax], "abcd"):
        panel_label(fig, ax, letter)
    style.save(fig, "fig-example4-model")


def plan_map(ax, x, y, values, cmap, norm, levels=None):
    contour = ax.tricontourf(x, y, values, levels=levels if levels is not None else 32, cmap=cmap, norm=norm)
    ax.plot(x, y, ".", color=style.INK, markersize=0.8)
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7.8, 12.2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return contour


def draw_fence(ax, reference, e: np.ndarray, values: np.ndarray, cmap, norm) -> None:
    """Dipole-dipole values of all lines as vertical pseudosection panels at their pseudodepths in a 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colormap = plt.get_cmap(cmap)
    half = SPACING / 4.0
    for line in range(len(e) // PER_LINE):
        index = np.arange(line * PER_LINE, (line + 1) * PER_LINE)
        rows = (reference.b >= 0) & np.isin(reference.a, index)
        a, b, m, n = (reference[k][rows].astype(int) for k in ("a", "b", "m", "n"))
        x = 0.25 * (e[a, 0] + e[b, 0] + e[m, 0] + e[n, 0]) / 1000.0
        z = pseudo.pseudodepth(m - b, SPACING, "dd")
        y = e[a[0], 1] / 1000.0
        quads = [[(xi - 0.05, y, zi - half), (xi + 0.05, y, zi - half), (xi + 0.05, y, zi + half),
                  (xi - 0.05, y, zi + half)] for xi, zi in zip(x, z)]
        ax.add_collection3d(Poly3DCollection(quads, facecolors=colormap(norm(values[rows])), edgecolors="none"))
    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(8.0, 12.0)
    ax.set_zlim(600.0, 0.0)
    ax.set_xticks([8, 10, 12])
    ax.set_yticks([8, 10, 12])
    ax.set_zticks([0, 200, 400, 600])
    ax.set_xlabel("x (km)", labelpad=-4)
    ax.set_ylabel("y (km)", labelpad=-4)
    ax.tick_params(pad=-2)
    ax.set_box_aspect((1.0, 1.0, 0.6), zoom=0.9)
    ax.view_init(elev=24, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(style.INK)
        axis.line.set_color(style.INK)
        axis._axinfo["grid"]["linewidth"] = 0.0


def responses_figure() -> None:
    e = electrodes()
    reference, _ = read(REFERENCE)
    dd = reference.b >= 0
    width, height = style.DOUBLE, 0.62 * style.DOUBLE
    fig = plt.figure(figsize=(width, height))
    rows = [(np.log10(reference.rho_a), style.RESISTIVITY_CMAP, r"$\log_{10}[\rho_a/(\Omega\,\mathrm{m})]$"),
            (reference.m_a * 1000.0, style.CHARGEABILITY_CMAP, r"$m_a$ (mV/V)")]
    h = 0.34
    w = h * height / width * 4.0 / 4.4
    letters = iter("abcdef")
    for row, (values, cmap, label) in enumerate(rows):
        y0 = 0.58 - row * 0.49
        norm = Normalize(*np.percentile(values[dd], [1, 99]))
        for column, level in enumerate(LEVELS):
            ax = fig.add_axes([0.07 + column * (w + 0.09), y0, w, h])
            x, y, mask = dd_points(reference, e, level)
            plan_map(ax, x, y, values[mask], cmap, norm)
            ax.set_title(depth_label(level))
            panel_label(fig, ax, next(letters))
        fence = fig.add_axes([0.07 + 2 * (w + 0.09) - 0.04, y0 - 0.06, 0.34, h + 0.1], projection="3d")
        draw_fence(fence, reference, e, values, cmap, norm)
        fence.text2D(1.1, 0.55, "Pseudodepth (m)", transform=fence.transAxes, rotation=90, va="center")
        fig.text(0.07 + 2 * (w + 0.09) - 0.04, y0 + h + 0.006, f"({next(letters)})", fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.965, y0, 0.012, h]), label=label,
                     extend="both")
    style.save(fig, "fig-example4-responses")


def errors_figure() -> None:
    e = electrodes()
    reference, _ = read(REFERENCE)
    width, height = style.DOUBLE, 0.62 * style.DOUBLE
    fig = plt.figure(figsize=(width, height))
    h = 0.35
    line_axes = [fig.add_axes([0.08 + i * 0.32, 0.57, 0.22, h]) for i in range(3)]
    cdf = fig.add_axes([0.08, 0.08, 0.22, h])
    w_map = h * height / width * 4.0 / 4.4
    maps = [fig.add_axes([0.42 + i * (w_map + 0.09), 0.08, w_map, h]) for i in range(2)]
    for key in style.BOUNDARIES:
        median, top, ip = [], [], []
        for extent in EXTENTS:
            table, _ = read(f"{key}-{extent}")
            rel = np.abs(table.rho_a / reference.rho_a - 1.0)
            median.append(np.median(rel))
            top.append(np.percentile(rel, 99))
            ip.append(np.percentile(np.abs(table.m_a - reference.m_a) * 1000.0, 99))
        for ax, values in zip(line_axes, (median, top, ip)):
            ax.plot(np.array(EXTENTS) / 1000.0, values, **style.BOUNDARIES[key])
        table, _ = read(f"{key}-{COMPARE}")
        rel = np.sort(np.abs(table.rho_a / reference.rho_a - 1.0))
        cdf.plot(rel, np.arange(1, len(rel) + 1) / len(rel), color=style.BOUNDARIES[key]["color"],
                 linestyle=style.BOUNDARIES[key]["linestyle"], label=style.BOUNDARIES[key]["label"])
    for ax, title, ylabel in zip(line_axes, (r"$\rho_a$, median", r"$\rho_a$, 99th percentile", r"$m_a$, 99th percentile"),
                                 ("Relative error", "Relative error", "Error (mV/V)")):
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Extent (km)")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    line_axes[0].legend(loc="lower left", fontsize=6)
    cdf.set_xscale("log")
    cdf.set_xlabel("Relative error")
    cdf.set_ylabel("Fraction of data")
    cdf.set_title(rf"$\rho_a$, {COMPARE / 1000:g} km")
    cdf.legend(loc="upper left", fontsize=6)
    level = LEVELS[1]
    x, y, rows = dd_points(reference, e, level)
    values = []
    for key in ("ds", "le"):
        table, _ = read(f"{key}-{COMPARE}")
        values.append(100.0 * (table.rho_a / reference.rho_a - 1.0)[rows])
    limit = max(np.abs(v).max() for v in values)
    norm = TwoSlopeNorm(0.0, -limit, limit)
    for ax, v, key in zip(maps, values, ("ds", "le")):
        plan_map(ax, x, y, v, style.ERROR_CMAP, norm, levels=np.linspace(-limit, limit, 33))
        ax.set_title(f"{style.BOUNDARIES[key]['label']}, {depth_label(level).lower()}")
    box = maps[1].get_position()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=style.ERROR_CMAP),
                 cax=fig.add_axes([box.x1 + 0.015, box.y0, 0.012, box.height]), label=r"$\Delta\rho_a$ (%)")
    for ax, letter in zip(line_axes + [cdf] + maps, "abcdef"):
        panel_label(fig, ax, letter)
    style.save(fig, "fig-example4-errors")


if __name__ == "__main__":
    style.apply()
    model_figure()
    metrics.require([RESULTS / REFERENCE] + [RESULTS / f"{m}-{e}" for m in style.BOUNDARIES for e in EXTENTS])
    responses_figure()
    errors_figure()
