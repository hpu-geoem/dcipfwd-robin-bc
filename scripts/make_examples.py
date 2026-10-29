"""Generate the inputs of Examples 1 and 2 of the paper (two-layer earth) under models/.

    python3 -I scripts/make_examples.py            # all four directories
    python3 -I scripts/make_examples.py ex01-basement-3d

ex01-basement-3d and ex02-basement-25d are Example 1 (3D and 2.5D), ex03-line48-3d and ex04-line48-25d Example 2. Each
directory gets the meshes of all outer extents, properties, a survey, the true layered background and a README. The
runs are defined in scripts/paper_runs.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcipio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

ELECTRODES = np.arange(-100.0, 100.0 + 1e-9, 20.0)  # 11 surface electrodes, 20 m apart
DOMAIN_EXTENTS = [500.0, 1000.0, 2000.0, 5000.0, 20000.0]


def line_survey(along: str) -> dcipio.Survey:
    electrodes = dcipio.line_electrodes(ELECTRODES, along=along)
    rows = np.concatenate([dcipio.pole_pole_rows(0, range(1, len(ELECTRODES))),
                           dcipio.dipole_dipole_rows(len(ELECTRODES), 6)])
    return dcipio.Survey(electrodes, rows)


def mesh_3d(extent: float, growth: float = 1.4) -> dcipio.TensorMesh:
    x = dcipio.padded_axis(120.0, 20.0, extent, growth)
    y = dcipio.padded_axis(40.0, 20.0, extent, growth)
    z = dcipio.padded_axis(100.0, 10.0, extent, growth, symmetric=False)  # Layer interfaces on 10 m planes
    return dcipio.TensorMesh([x, y, z])


def mesh_2d(extent: float, growth: float = 1.4) -> dcipio.TensorMesh:
    y = dcipio.padded_axis(120.0, 10.0, extent, growth)
    z = dcipio.padded_axis(100.0, 10.0, extent, growth, symmetric=False)
    return dcipio.TensorMesh([y, z])


def write_readme(directory: Path, text: str) -> None:
    (directory / "README.md").write_text(text.strip() + "\n")


def basement(dim: int) -> None:
    """Conductive cover over a resistive basement: the case where the homogeneous mixed condition fails."""
    name = f"ex0{1 if dim == 3 else 2}-basement-{'3d' if dim == 3 else '25d'}"
    directory = MODELS / name
    directory.mkdir(parents=True, exist_ok=True)
    rho1, rho2, h = 100.0, 1000.0, 50.0
    cells = []
    for extent in DOMAIN_EXTENTS:
        mesh = mesh_3d(extent) if dim == 3 else mesh_2d(extent)
        depth_axis = dim - 1
        mesh.assign(lambda *c: np.where(c[depth_axis] < h, 1, 2))
        mesh.write_mfem(directory / f"model-{int(extent)}.mesh")
        cells.append(int(np.prod(mesh.shape)))
    eta2 = 0.1  # Chargeable basement for the IP pass
    dcipio.write_properties(directory / "model.prop", {1: (rho1, 0.0), 2: (rho2, eta2)})
    dcipio.write_background(directory / "background.txt", [(0.0, rho1, 0.0), (h, rho2, eta2)])
    line_survey("x" if dim == 3 else "y").write(directory / "survey.txt")
    sizes = ", ".join(f"{int(e)} m: {c} cells" for e, c in zip(DOMAIN_EXTENTS, cells))
    write_readme(directory, f"""
# {name} (Example 1 of the paper, {'3D' if dim == 3 else '2.5D'})

{'3D' if dim == 3 else '2.5D (y-z section, strike along x)'} two-layer earth: {rho1:g} ohm-m cover, {h:g} m thick,
over a {rho2:g} ohm-m basement with integral chargeability {eta2:g} (L = h rho2 / rho1 = {h * rho2 / rho1:g} m; the homogeneous mixed condition needs
R of about 10 L, Section 2 of the paper). The same core mesh is padded to several outer extents: {sizes}.

Eleven surface electrodes from -100 to 100 m (20 m spacing) on the {'x' if dim == 3 else 'y'} axis record pole-pole data
from electrode 0 and dipole-dipole data with n = 1..6. Analytic responses: N-layer quadrature of scripts/analytic.py.
Runs: group example1 of scripts/paper_runs.py; background variants of the robustness checks in backgrounds/
(scripts/make_backgrounds.py).
""")


def line48(dim: int) -> None:
    """48 electrodes, 20 m apart, over the ex01 two-layer earth: the reference-source study (E4)."""
    name = f"ex0{3 if dim == 3 else 4}-line48-{'3d' if dim == 3 else '25d'}"
    directory = MODELS / name
    directory.mkdir(parents=True, exist_ok=True)
    rho1, rho2, h = 100.0, 1000.0, 50.0
    positions = np.arange(-470.0, 470.0 + 1e-9, 20.0)
    extents = [1000.0, 2000.0, 20000.0]
    cells = []
    for extent in extents:
        if dim == 3:
            mesh = dcipio.TensorMesh([dcipio.padded_axis(500.0, 20.0, extent, 1.4), dcipio.padded_axis(40.0, 20.0, extent, 1.4),
                                      dcipio.padded_axis(100.0, 10.0, extent, 1.4, symmetric=False)])
        else:
            mesh = dcipio.TensorMesh([dcipio.padded_axis(500.0, 10.0, extent, 1.4),
                                      dcipio.padded_axis(100.0, 10.0, extent, 1.4, symmetric=False)])
        mesh.assign(lambda *c: np.where(c[dim - 1] < h, 1, 2))
        mesh.write_mfem(directory / f"model-{int(extent)}.mesh")
        cells.append(int(np.prod(mesh.shape)))
    dcipio.write_properties(directory / "model.prop", {1: (rho1, 0.0), 2: (rho2, 0.0)})
    dcipio.write_background(directory / "background.txt", [(0.0, rho1, 0.0), (h, rho2, 0.0)])
    electrodes = dcipio.line_electrodes(positions, along="x" if dim == 3 else "y")
    n = len(positions)
    rows = np.concatenate([dcipio.pole_pole_rows(0, range(1, n)), dcipio.pole_pole_rows(n - 1, range(0, n - 1)),
                           dcipio.dipole_dipole_rows(n, 8)])
    dcipio.Survey(electrodes, rows).write(directory / "survey.txt")
    sizes = ", ".join(f"{int(e)} m: {c} cells" for e, c in zip(extents, cells))
    current = np.unique(rows[:, :2][rows[:, :2] >= 0]).size
    write_readme(directory, f"""
# {name} (Example 2 of the paper, {'3D' if dim == 3 else '2.5D'})

{'3D' if dim == 3 else '2.5D'} two-layer earth as ex01 ({rho1:g} ohm-m, {h:g} m over {rho2:g} ohm-m); 48 surface electrodes from
-470 to 470 m (20 m spacing) on the {'x' if dim == 3 else 'y'} axis; pole-pole rows from both line ends and dipole-dipole rows
with n = 1..8 ({len(rows)} rows; {current} of the {n} electrodes carry current). Core cells {'20 m x 20 m x 10 m' if dim == 3 else '10 m x 10 m'};
outer extents {sizes}. Runs: group example2 of scripts/paper_runs.py.
""")


EXAMPLES = {
    "ex01-basement-3d": lambda: basement(3),
    "ex02-basement-25d": lambda: basement(2),
    "ex03-line48-3d": lambda: line48(3),
    "ex04-line48-25d": lambda: line48(2),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or EXAMPLES:
        EXAMPLES[name]()
        print(f"wrote models/{name}")
