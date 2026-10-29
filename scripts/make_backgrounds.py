"""Background variants of the robustness checks (the true models are unchanged; only the background that defines the
boundary condition differs).

    python3 -I scripts/make_backgrounds.py

Example 1 (two-layer earth: 50 m of 100 ohm-m over 1000 ohm-m with chargeability 0.1), in
models/ex0{1,2}-basement-*/backgrounds/: cover thickness 40 or 60 m (h40, h60), cover resistivity 80 or 125 ohm-m
(rho1-80, rho1-125), basement resistivity 500 or 2000 ohm-m (rho2-500, rho2-2000).
Example 3 (Overthrust section, twelve 400 m layers of lateral means), in models/e8-overthrust/octree-section/backgrounds/:
all resistivities times 0.8 or 1.25 (rho080, rho125), all layer tops times 0.8 or 1.2 (depth080, depth120), the
column of the westernmost 100 m of the core instead of the lateral mean (edge), and three layers of lateral means
over 0-1.2, 1.2-2.8 and below 2.8 km (coarse3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcipio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "models" / "e8-overthrust" / "octree-section"


def example1() -> None:
    variants = {"h40": (100.0, 40.0, 1000.0), "h60": (100.0, 60.0, 1000.0), "rho1-80": (80.0, 50.0, 1000.0),
                "rho1-125": (125.0, 50.0, 1000.0), "rho2-500": (100.0, 50.0, 500.0), "rho2-2000": (100.0, 50.0, 2000.0)}
    for model in ("ex01-basement-3d", "ex02-basement-25d"):
        directory = ROOT / "models" / model / "backgrounds"
        directory.mkdir(exist_ok=True)
        for name, (rho1, thickness, rho2) in variants.items():
            dcipio.write_background(directory / f"{name}.txt", [(0.0, rho1, 0.0), (thickness, rho2, 0.1)])


def section_voxels() -> tuple[np.ndarray, np.ndarray, float]:
    header = {}
    for row in (SECTION / "voxels.txt").read_text().splitlines():
        if row and not row.startswith("#"):
            key, *values = row.split()
            header[key] = values
    nx, nz = (int(v) for v in header["shape"])
    data = np.fromfile(SECTION / "voxels.bin", dtype="<f8")
    log_rho = data[:nx * nz].reshape(nz, nx)
    eta = data[nx * nz:].reshape(nz, nx)
    return log_rho, eta, float(header["spacing"][0])


def means(log_rho: np.ndarray, eta: np.ndarray, tops: list[float], spacing: float) -> list[tuple[float, float, float]]:
    rows = [int(round(top / spacing)) for top in tops] + [log_rho.shape[0]]
    return [(top, float(10.0 ** log_rho[first:last].mean()), float(eta[first:last].mean()))
            for top, first, last in zip(tops, rows[:-1], rows[1:])]


def example3() -> None:
    directory = SECTION / "backgrounds"
    directory.mkdir(exist_ok=True)
    true = [(float(top), float(rho), float(eta)) for top, rho, eta in
            np.loadtxt(SECTION / "background.txt", comments="#", ndmin=2)]
    for name, scale in (("rho080", 0.8), ("rho125", 1.25)):
        dcipio.write_background(directory / f"{name}.txt", [(t, r * scale, e) for t, r, e in true])
    for name, scale in (("depth080", 0.8), ("depth120", 1.2)):
        dcipio.write_background(directory / f"{name}.txt", [(t * scale, r, e) for t, r, e in true])
    log_rho, eta, spacing = section_voxels()
    tops = [t for t, _, _ in true]
    columns = int(round(100.0 / spacing))
    dcipio.write_background(directory / "edge.txt", means(log_rho[:, :columns], eta[:, :columns], tops, spacing))
    dcipio.write_background(directory / "coarse3.txt", means(log_rho, eta, [0.0, 1200.0, 2800.0], spacing))
    # The twelve-layer lateral means must reproduce the true background file.
    check = means(log_rho, eta, tops, spacing)
    assert all(abs(a[1] / b[1] - 1.0) < 1e-8 for a, b in zip(check, true)), "voxels do not match background.txt"


if __name__ == "__main__":
    example1()
    example3()
    print("Wrote the background variants of examples 1 and 3.")
