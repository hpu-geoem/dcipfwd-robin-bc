"""Inputs of the octree (quadtree) Overthrust examples 3 and 4 for code/apps/octree_mesh.cc.

    python3 -I scripts/make_overthrust_octree.py section   # models/e8-overthrust/octree-section/
    python3 -I scripts/make_overthrust_octree.py volume    # models/e8-overthrust/octree-volume/

Each directory receives
- voxels.txt/.bin: log10 resistivity and chargeability of the 25 m Overthrust cells of the core (x fastest);
- background.txt: lateral means of the core over 400 m depth intervals (log mean for resistivity);
- root-<extent>.mesh: conforming root mesh, 400 m cells in the core (attribute 1) and padding cells growing by about 1.3
  (attribute 2) to exactly the outer extent beyond the core;
- survey*.txt and octree-options.txt (the refinement options passed to octree_mesh).

The core spans x (and y) from 7.6 to 12.4 km and depths from 0 to 4.8 km; the 25 m model ends at 4.675 km and its
deepest row is repeated below. Property mappings and the source file are those of scripts/make_overthrust.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcipio  # noqa: E402
import make_overthrust as mo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "e8-overthrust"
CORE = (7600.0, 12400.0)
DEPTH = 4800.0
ROOT_CELL = 400.0
GROWTH = 1.3
EXTENTS = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 200000.0, 400000.0]
LINE = np.arange(8000.0, 12000.0 + 1e-9, 100.0)
LINE_OFFSETS = np.arange(8000.0, 12000.0 + 1e-9, 500.0)
OPTIONS = {
    "section": "-tol-rho 0.1 -tol-eta 0.02 -min-size 1e30:25 -surface-box 8000,12000 -surface-size 25",
    "volume": "-tol-rho 0.1 -tol-eta 0.02 -min-size 1000:25,2500:50,1e30:100 -surface-box 8000,12000,8000,12000 "
              "-surface-size 50",
}


def slowness(section: bool) -> np.ndarray:
    """Squared slowness (float32, s^2/km^2) of the core on the 25 m grid, [z, y, x], without the water layer."""
    import h5py

    ix = slice(int(round(CORE[0] / mo.DX)), int(round(CORE[1] / mo.DX)))
    iy = slice(int(round(10000.0 / mo.DX)), int(round(10000.0 / mo.DX)) + 1) if section else ix
    with h5py.File(mo.SOURCE, "r") as f:
        return np.asarray(f["m"][mo.WATER_CELLS:, iy, ix])


def voxels_from_slowness(m: np.ndarray, section: bool) -> tuple[np.ndarray, np.ndarray]:
    """log10(rho) and eta of the core on the 25 m grid, [z, y, x] (3D) or [z, x] (the section y = 10 km)."""
    v = 1.0 / np.sqrt(np.asarray(m, dtype=float))
    log_rho, eta = np.log10(mo.rho_of(v)), mo.eta_of(v)
    if section:
        log_rho, eta = log_rho[:, 0, :], eta[:, 0, :]
    rows = int(round(DEPTH / mo.DX)) - log_rho.shape[0]
    assert rows >= 0
    return (np.concatenate([log_rho, np.repeat(log_rho[-1:], rows, axis=0)]),
            np.concatenate([eta, np.repeat(eta[-1:], rows, axis=0)]))


def voxels(section: bool) -> tuple[np.ndarray, np.ndarray]:
    return voxels_from_slowness(slowness(section), section)


def write_voxels(directory: Path, log_rho: np.ndarray, eta: np.ndarray) -> None:
    dim = log_rho.ndim
    shape = log_rho.shape[::-1]  # Axis 0 (x, or y of the section) varies fastest.
    origin = [CORE[0]] * (dim - 1) + [0.0]
    (directory / "voxels.txt").write_text(
        "# Overthrust core: log10 resistivity then chargeability as float64, axis 0 fastest; depth is the last axis\n"
        f"dimension {dim}\norigin {' '.join(f'{v:g}' for v in origin)}\nspacing {mo.DX:g}\n"
        f"shape {' '.join(str(n) for n in shape)}\n")
    with open(directory / "voxels.bin", "wb") as stream:
        stream.write(np.ascontiguousarray(log_rho, dtype="<f8").tobytes())
        stream.write(np.ascontiguousarray(eta, dtype="<f8").tobytes())


def background(log_rho: np.ndarray, eta: np.ndarray) -> list[tuple[float, float, float]]:
    """Lateral log means of resistivity and means of chargeability over 400 m depth intervals."""
    per_layer = int(round(ROOT_CELL / mo.DX))
    layers = []
    for row in range(0, log_rho.shape[0], per_layer):
        layers.append((row * mo.DX, float(10.0 ** log_rho[row:row + per_layer].mean()),
                       float(eta[row:row + per_layer].mean())))
    return layers


def padding(extent: float) -> np.ndarray:
    """Widths of padding cells growing by GROWTH from ROOT_CELL, scaled uniformly to sum to the extent; the cell count
    is the one whose unscaled sum is closest to the extent in log scale."""
    widths = ROOT_CELL * GROWTH ** np.arange(1, 80)
    count = int(np.argmin(np.abs(np.log(np.cumsum(widths) / extent)))) + 1
    return widths[:count] * extent / widths[:count].sum()


def padded(nodes: np.ndarray, extent: float, lower: bool = True) -> np.ndarray:
    widths = padding(extent)
    upper = nodes[-1] + np.cumsum(widths)
    below = nodes[0] - np.cumsum(widths)[::-1] if lower else np.array([])
    return np.concatenate([below, nodes, upper])


def root_mesh(dim: int, extent: float) -> dcipio.TensorMesh:
    core = mo.coarse_axis(*CORE, ROOT_CELL)
    depth = mo.coarse_axis(0.0, DEPTH, ROOT_CELL)
    axes = [padded(core, extent) for _ in range(dim - 1)] + [padded(depth, extent, lower=False)]
    mesh = dcipio.TensorMesh(axes)
    mesh.assign(lambda *c: np.where(np.logical_and.reduce(
        [(x > CORE[0]) & (x < CORE[1]) for x in c[:-1]] + [c[-1] < DEPTH]), 1, 2))
    return mesh


def line_survey(along: str, offset: float = 0.0) -> dcipio.Survey:
    return mo.survey(LINE, along, offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["section", "volume"])
    parser.add_argument("--extents", type=float, nargs="*", default=EXTENTS)
    parser.add_argument("--slowness", type=Path, help="rebuild voxels.{txt,bin} only, from a float32 [z, y, x] file "
                        "written by --write-slowness (for machines without the source model)")
    parser.add_argument("--write-slowness", type=Path, help="write the squared slowness of the core and exit")
    args = parser.parse_args()
    section = args.part == "section"
    directory = BASE / f"octree-{args.part}"
    directory.mkdir(parents=True, exist_ok=True)
    if args.write_slowness:
        slowness(section).astype("<f4").tofile(args.write_slowness)
        return
    if args.slowness:
        n = 1 if section else round((CORE[1] - CORE[0]) / mo.DX)
        m = np.fromfile(args.slowness, dtype="<f4").reshape(-1, n, round((CORE[1] - CORE[0]) / mo.DX))
        write_voxels(directory, *voxels_from_slowness(m, section))
        return
    log_rho, eta = voxels(section)
    write_voxels(directory, log_rho, eta)
    layers = background(log_rho, eta)
    dcipio.write_background(directory / "background.txt", layers)
    dim = 2 if section else 3
    sizes = {}
    for extent in args.extents:
        mesh = root_mesh(dim, extent)
        mesh.write_mfem(directory / f"root-{int(extent)}.mesh")
        sizes[str(int(extent))] = {"root_shape": list(mesh.shape),
                                   "outer": [[float(a[0]), float(a[-1])] for a in mesh.axes]}
    if section:
        line_survey("y").write(directory / "survey.txt")
    else:
        lines = [line_survey("x", offset) for offset in LINE_OFFSETS]
        electrodes = np.concatenate([line.electrodes for line in lines])
        rows = np.concatenate([np.where(line.rows >= 0, line.rows + i * len(LINE), -1) for i, line in enumerate(lines)])
        dcipio.Survey(electrodes, rows).write(directory / "survey.txt")
        line_survey("x", 10000.0).write(directory / "survey-central.txt")
    (directory / "octree-options.txt").write_text(OPTIONS[args.part] + "\n")
    (directory / "sizes.json").write_text(json.dumps({
        "core": list(CORE), "depth": DEPTH, "root_cell": ROOT_CELL, "growth": GROWTH, "voxel": mo.DX,
        "voxel_shape": list(log_rho.shape), "background": layers, "extents": sizes,
        "line_offsets": LINE_OFFSETS.tolist() if not section else [10000.0], "electrodes_per_line": len(LINE)},
        indent=1))
    print(args.part, "voxels", log_rho.shape, "background", [(t, round(r, 1), round(e, 3)) for t, r, e in layers])


if __name__ == "__main__":
    main()
