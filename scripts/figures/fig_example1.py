"""Figures for example 1 (two-layer earth, 3D and 2.5D).

- fig-example1-responses: pole-pole and dipole-dipole apparent resistivity and apparent chargeability of the 3D and
  2.5D solutions on the 0.5 km domain against the analytic two-layer responses;
- fig-example1: largest truncation error against the outer extent (reference: LE at 20 km on the same core mesh).
Data: results/example1-3d and results/example1-25d (scripts/paper_runs.py, group example1); analytic responses from
scripts/analytic.py. Also writes figures/data/example1.csv and figures/data/example1-responses.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analytic  # noqa: E402
import dcipio  # noqa: E402
import style  # noqa: E402

EXTENTS = [500, 1000, 2000, 5000]
VARIANTS = ["ds", "de", "ls", "le", "li"]
CASES = [("example1-3d", "3-D"), ("example1-25d", "2.5-D")]
RHO, THICKNESS, ETA = [100.0, 1000.0], [50.0], 0.1


def errors(directory: Path, variant: str, extent: int, reference):
    table, _ = dcipio.read_result(directory / f"{variant}-{extent}")
    pole = table.b < 0
    rel = np.abs(table.dV / reference.dV - 1.0)
    ip = np.abs(table.m_a - reference.m_a) * 1000.0
    return rel[pole].max(), rel[~pole].max(), ip.max()


def electrodes() -> np.ndarray:
    lines = [ln.split("#")[0].strip() for ln in (style.ROOT / "models" / "ex01-basement-3d" / "survey.txt").read_text()
             .splitlines()]
    lines = [ln for ln in lines if ln]
    n = int(lines[0])
    return np.array([[float(v) for v in ln.split()] for ln in lines[1:1 + n]])


def analytic_responses(table, x: np.ndarray):
    """Apparent resistivity and chargeability of the rows of a response table on the two-layer earth."""
    offsets = np.arange(20.0, 201.0, 20.0)
    dc, _ = analytic.layered_surface_potential(RHO, THICKNESS, offsets)
    inst, _ = analytic.layered_surface_potential([RHO[0], RHO[1] * (1.0 - ETA)], THICKNESS, offsets)

    def potential(values, i, j):
        if i < 0 or j < 0:
            return 0.0
        return values[int(round(abs(x[i] - x[j]) / 20.0)) - 1]

    def difference(values):
        return np.array([potential(values, a, m) - potential(values, a, n) - potential(values, b, m) + potential(values, b, n)
                         for a, b, m, n in zip(*(table[k].astype(int) for k in ("a", "b", "m", "n")))])

    dv, dv_inst = difference(dc), difference(inst)
    return dv * table.K, 1.0 - dv_inst / dv


def responses_figure() -> None:
    x = electrodes()[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 0.33 * style.DOUBLE), gridspec_kw={"wspace": 0.38})
    rows = ["case,variant,a,b,m,n,rho_a,m_a,rho_a_analytic,m_a_analytic"]
    reference, _ = dcipio.read_result(style.ROOT / "results" / "example1-3d" / "le-500")
    rho_a, m_a = analytic_responses(reference, x)
    pole = reference.b < 0
    offset_pp = np.abs(x[reference.m.astype(int)] - x[reference.a.astype(int)])
    centre = np.abs(0.5 * (x[reference.m.astype(int)] + x[reference.n.astype(int)])
                    - 0.5 * (x[reference.a.astype(int)] + x[np.maximum(reference.b.astype(int), 0)]))
    order = np.argsort(offset_pp[pole])
    axes[0].plot(offset_pp[pole][order], rho_a[pole][order], color=style.INK, label="Analytic")
    dd_levels = np.unique(centre[~pole])
    dd_rho = [rho_a[~pole][centre[~pole] == c][0] for c in dd_levels]
    dd_ma = [m_a[~pole][centre[~pole] == c][0] for c in dd_levels]
    axes[1].plot(dd_levels, dd_rho, color=style.INK, label="Analytic")
    axes[2].plot(offset_pp[pole][order], 1000.0 * m_a[pole][order], color=style.INK, label="Analytic, PP")
    axes[2].plot(dd_levels, 1000.0 * np.array(dd_ma), color=style.INK, linestyle="--", label="Analytic, DD")
    series = [("example1-3d", "le", "LE, 3-D", style.RED, "^"),
              ("example1-25d", "le", "LE, 2.5-D", style.RED, "D"),
              ("example1-3d", "ds", "DS, 3-D", style.BLUE, "o")]
    for case, variant, label, color, marker in series:
        table, _ = dcipio.read_result(style.ROOT / "results" / case / f"{variant}-500")
        p = table.b < 0
        kwargs = dict(color=color, marker=marker, linestyle="none", markerfacecolor="none", markersize=4)
        axes[0].plot(offset_pp[p], table.rho_a[p], label=label, **kwargs)
        axes[1].plot(centre[~p], table.rho_a[~p], label=label, **kwargs)
        axes[2].plot(offset_pp[p], 1000.0 * table.m_a[p], label=label, **kwargs)
        axes[2].plot(centre[~p], 1000.0 * table.m_a[~p], **kwargs)
        for row, ra, ma, rr, mr in zip(table, table.rho_a, table.m_a, rho_a, m_a):
            rows.append(f"{case},{variant},{int(row.a)},{int(row.b)},{int(row.m)},{int(row.n)},{ra:.6e},{ma:.6e},"
                        f"{rr:.6e},{mr:.6e}")
    axes[0].set_title("Pole-pole")
    axes[0].set_xlabel("Offset AM (m)")
    axes[0].set_ylabel(r"$\rho_a$ ($\Omega$m)")
    axes[1].set_title("Dipole-dipole")
    axes[1].set_xlabel("Dipole-centre distance (m)")
    axes[1].set_ylabel(r"$\rho_a$ ($\Omega$m)")
    axes[2].set_title("Apparent chargeability")
    axes[2].set_xlabel("Offset (m)")
    axes[2].set_ylabel(r"$m_a$ (mV/V)")
    axes[0].legend(loc="upper left")
    axes[2].legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=6.5)
    style.label_panels(axes)
    style.save(fig, "fig-example1-responses")
    (style.ROOT / "figures" / "data" / "example1-responses.csv").write_text("\n".join(rows) + "\n")


def errors_figure() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE, 0.6 * style.DOUBLE),
                             gridspec_kw={"wspace": 0.42, "hspace": 0.55, "top": 0.88})
    rows = ["case,variant,extent_m,pole_rel,dipole_rel,ma_abs_mVV"]
    for row, (case, label) in enumerate(CASES):
        directory = style.ROOT / "results" / case
        reference, _ = dcipio.read_result(directory / "le-20000")
        for variant in VARIANTS:
            values = np.array([errors(directory, variant, e, reference) for e in EXTENTS])
            for extent, v in zip(EXTENTS, values):
                rows.append(f"{case},{variant},{extent},{v[0]:.4e},{v[1]:.4e},{v[2]:.4e}")
            for column in range(3):
                axes[row, column].plot(np.array(EXTENTS) / 1000.0, values[:, column], **style.BOUNDARIES[variant])
        for column, title in enumerate(["pole-pole", "dipole-dipole", r"$m_a$"]):
            ax = axes[row, column]
            ax.set_title(f"{label}, {title}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.5, 1, 2, 5])
            ax.set_xticklabels(["0.5", "1", "2", "5"])
            ax.set_xlabel("Extent (km)")
            ax.set_ylabel("Error (mV/V)" if column == 2 else "Relative error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995))
    style.label_panels(axes.ravel())
    style.save(fig, "fig-example1")
    (style.ROOT / "figures" / "data" / "example1.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    style.apply()
    responses_figure()
    errors_figure()
