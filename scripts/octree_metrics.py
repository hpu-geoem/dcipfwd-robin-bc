"""Representation error of the octree Overthrust models against their 25 m voxel models, and of the 100 m tensor mesh
of the earlier design for comparison; writes figures/data/octree-metrics.json.

    python3 -I scripts/octree_metrics.py     (needs models/e8-overthrust/octree-*/model-500.cells from octree_mesh)

The error is the root mean square over voxels of log10(rho) and eta minus the value of the cell that contains the voxel,
over the whole core and over its top 1 km.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "e8-overthrust"


def voxels(directory: Path):
    header = {}
    for line in (directory / "voxels.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            key, *values = line.split()
            header[key] = [float(v) for v in values]
    shape = [int(n) for n in header["shape"]]
    data = np.fromfile(directory / "voxels.bin", dtype="<f8")
    count = int(np.prod(shape))
    reshape = tuple(shape[::-1])  # [z, (y,) x]
    return header["origin"], header["spacing"][0], data[:count].reshape(reshape), data[count:].reshape(reshape)


def rms(cell_rho, cell_eta, rho, eta, rows):
    return (float(np.sqrt(np.mean((cell_rho[:rows] - rho[:rows]) ** 2))),
            float(np.sqrt(np.mean((cell_eta[:rows] - eta[:rows]) ** 2))))


def octree_fields(cells: np.ndarray, origin, spacing, shape):
    """Paint the core leaf values onto the voxel grid."""
    dim = len(shape)
    core = cells[cells[:, -1] == 1]
    rho = np.empty(shape[::-1])
    eta = np.empty(shape[::-1])
    index = np.rint((core[:, :2 * dim] - np.tile(origin, 2)) / spacing).astype(int)
    for row, value_rho, value_eta in zip(index, core[:, 2 * dim], core[:, 2 * dim + 1]):
        lower, upper = row[:dim], row[dim:]
        box = tuple(slice(lower[a], upper[a]) for a in reversed(range(dim)))
        rho[box] = value_rho
        eta[box] = value_eta
    return rho, eta, core


def tensor_fields(rho, eta, horizontal_cells: int, z_nodes: np.ndarray, spacing: float):
    """Block means on the earlier tensor mesh: horizontal cells of horizontal_cells voxels, vertical nodes z_nodes."""
    rows = np.rint(z_nodes / spacing).astype(int)
    out_rho, out_eta = np.empty_like(rho), np.empty_like(eta)
    count = 0
    for top, bottom in zip(rows[:-1], rows[1:]):
        for index in np.ndindex(*[n // horizontal_cells for n in rho.shape[1:]]):
            box = (slice(top, bottom),) + tuple(slice(i * horizontal_cells, (i + 1) * horizontal_cells) for i in index)
            out_rho[box] = rho[box].mean()
            out_eta[box] = eta[box].mean()
            count += 1
    return out_rho, out_eta, count


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_overthrust as mo

    z_nodes = np.concatenate([mo.depth_nodes(), [4800.0]])
    result = {}
    for part, horizontal in (("section", 2), ("volume", 4)):  # earlier designs: 50 m (section) and 100 m (volume)
        directory = BASE / f"octree-{part}"
        origin, spacing, rho, eta = voxels(directory)
        cells = np.loadtxt(directory / "model-500.cells")
        o_rho, o_eta, core = octree_fields(cells, origin, spacing, rho.shape[::-1])
        t_rho, t_eta, t_count = tensor_fields(rho, eta, horizontal, z_nodes, spacing)
        top = int(round(1000.0 / spacing))
        widths = np.round(core[:, rho.ndim] - core[:, 0]).astype(int)
        result[part] = {
            "octree_core_cells": int(len(core)),
            "octree_cells_by_width_m": {str(w): int((widths == w).sum()) for w in np.unique(widths)},
            "octree_rms_log10_rho_eta_all": rms(o_rho, o_eta, rho, eta, None),
            "octree_rms_log10_rho_eta_top1km": rms(o_rho, o_eta, rho, eta, top),
            "tensor_core_cells": int(t_count),
            "tensor_horizontal_m": horizontal * spacing,
            "tensor_rms_log10_rho_eta_all": rms(t_rho, t_eta, rho, eta, None),
            "tensor_rms_log10_rho_eta_top1km": rms(t_rho, t_eta, rho, eta, top),
        }
        print(part, json.dumps(result[part]))
    (ROOT / "figures" / "data" / "octree-metrics.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
