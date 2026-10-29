"""Inputs and outputs of dcipfwd: tensor meshes in MFEM format, properties, surveys, backgrounds and results.

Coordinates follow dcipfwd: z points down; a 3D mesh uses (x, y, z) and a 2D mesh is the (y, z) section of a
2.5D model with strike along x. Meshes cover the ground only; attributes start at 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

def padded_axis(core_half_width: float, core_cell: float, padding_extent: float, growth: float = 1.3,
                symmetric: bool = True) -> np.ndarray:
    """Return node coordinates: uniform cells of core_cell over |u| <= core_half_width, then cells growing by
    `growth` until the axis reaches padding_extent. With symmetric=False the axis starts at 0 (a depth axis)."""
    core = np.arange(0.0, core_half_width + 0.5 * core_cell, core_cell)
    nodes = list(core)
    cell = core_cell
    while nodes[-1] < padding_extent:
        cell *= growth
        nodes.append(nodes[-1] + cell)
    positive = np.array(nodes)
    if not symmetric:
        return positive
    return np.concatenate([-positive[:0:-1], positive])


@dataclass
class TensorMesh:
    """Tensor-product hexahedral (3D) or quadrilateral (2D) mesh given by its node lines."""

    axes: list[np.ndarray]
    attributes: np.ndarray = field(default=None)  # One attribute per cell, C order over the axes

    @property
    def dim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis) - 1 for axis in self.axes)

    def centers(self) -> list[np.ndarray]:
        """Return cell-center coordinate arrays shaped like the cells."""
        mids = [0.5 * (axis[1:] + axis[:-1]) for axis in self.axes]
        return list(np.meshgrid(*mids, indexing="ij"))

    def assign(self, attribute_of: Callable[..., np.ndarray]) -> "TensorMesh":
        """Set attributes from a vectorized function of the cell-center coordinates."""
        self.attributes = np.asarray(attribute_of(*self.centers()), dtype=int)
        assert self.attributes.shape == self.shape and self.attributes.min() >= 1
        return self

    def write_mfem(self, path: str | Path) -> None:
        """Write the mesh in MFEM v1.0 format with its outer boundary (attribute 1)."""
        assert self.attributes is not None, "Assign attributes before writing."
        n = [len(axis) for axis in self.axes]
        index = np.arange(np.prod(n)).reshape(n)
        lines = ["MFEM mesh v1.0", "", "dimension", str(self.dim), "", "elements"]
        if self.dim == 3:
            v = [index[:-1, :-1, :-1], index[1:, :-1, :-1], index[1:, 1:, :-1], index[:-1, 1:, :-1],
                 index[:-1, :-1, 1:], index[1:, :-1, 1:], index[1:, 1:, 1:], index[:-1, 1:, 1:]]
            cells = np.stack([a.ravel() for a in v], axis=1)
            lines.append(str(len(cells)))
            lines += [f"{a} 5 " + " ".join(map(str, c)) for a, c in zip(self.attributes.ravel(), cells)]
            faces = []
            for axis in range(3):
                for side in (0, -1):
                    sl = [slice(None)] * 3
                    sl[axis] = side
                    plane = index[tuple(sl)]
                    quad = [plane[:-1, :-1], plane[1:, :-1], plane[1:, 1:], plane[:-1, 1:]]
                    faces.append(np.stack([q.ravel() for q in quad], axis=1))
            faces = np.concatenate(faces)
            lines += ["", "boundary", str(len(faces))] + ["1 3 " + " ".join(map(str, f)) for f in faces]
        else:
            v = [index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]]
            cells = np.stack([a.ravel() for a in v], axis=1)
            lines.append(str(len(cells)))
            lines += [f"{a} 3 " + " ".join(map(str, c)) for a, c in zip(self.attributes.ravel(), cells)]
            segments = []
            for axis in range(2):
                for side in (0, -1):
                    line = index[side, :] if axis == 0 else index[:, side]
                    segments.append(np.stack([line[:-1], line[1:]], axis=1))
            segments = np.concatenate(segments)
            lines += ["", "boundary", str(len(segments))] + ["1 1 " + " ".join(map(str, s)) for s in segments]
        grids = np.meshgrid(*self.axes, indexing="ij")
        vertices = np.stack([g.ravel() for g in grids], axis=1)
        lines += ["", "vertices", str(len(vertices)), str(self.dim)]
        lines += [" ".join(f"{x:.10g}" for x in row) for row in vertices]
        Path(path).write_text("\n".join(lines) + "\n")


def write_properties(path: str | Path, properties: dict[int, tuple[float, float]]) -> None:
    """Write rows 'attribute resistivity chargeability'."""
    rows = ["# attribute resistivity(ohm-m) chargeability(V/V)"]
    rows += [f"{a} {rho:.10g} {eta:.10g}" for a, (rho, eta) in sorted(properties.items())]
    Path(path).write_text("\n".join(rows) + "\n")


def write_background(path: str | Path, layers: Sequence[tuple[float, float, float]]) -> None:
    """Write rows 'top resistivity chargeability' of a layered background."""
    rows = ["# top(m, z down) resistivity(ohm-m) chargeability(V/V)"]
    rows += [f"{top:.10g} {rho:.10g} {eta:.10g}" for top, rho, eta in layers]
    Path(path).write_text("\n".join(rows) + "\n")


@dataclass
class Survey:
    electrodes: np.ndarray  # (n, 3) positions x, y, z
    rows: np.ndarray        # (m, 4) indices a, b, m, n; -1 = pole

    def write(self, path: str | Path) -> None:
        lines = ["# electrodes: x y z (z down)", str(len(self.electrodes))]
        lines += [" ".join(f"{c:.10g}" for c in e) for e in self.electrodes]
        lines += ["# measurements: a b m n (-1 = pole at infinity)", str(len(self.rows))]
        lines += [" ".join(str(int(i)) for i in r) for r in self.rows]
        Path(path).write_text("\n".join(lines) + "\n")


def line_electrodes(positions: Sequence[float], along: str = "x", depth: float = 0.0) -> np.ndarray:
    """Return electrodes on a surface line along x (3D) or y (2.5D)."""
    e = np.zeros((len(positions), 3))
    e[:, 0 if along == "x" else 1] = positions
    e[:, 2] = depth
    return e


def pole_pole_rows(source: int, receivers: Sequence[int]) -> np.ndarray:
    return np.array([[source, -1, r, -1] for r in receivers], dtype=int)


def dipole_dipole_rows(n_electrodes: int, max_n: int) -> np.ndarray:
    """Return dipole-dipole rows (A, B, M, N) = (i, i+1, i+1+k, i+2+k) for separations k = 1..max_n."""
    rows = []
    for k in range(1, max_n + 1):
        for i in range(n_electrodes - 2 - k):
            rows.append([i, i + 1, i + 1 + k, i + 2 + k])
    return np.array(rows, dtype=int)


def read_result(prefix: str | Path) -> tuple[np.ndarray, dict]:
    """Return the rows of <prefix>.dat as a structured array and the run report of <prefix>.json."""
    prefix = str(prefix)
    data = np.loadtxt(prefix + ".dat", comments="#", ndmin=2)
    names = ["a", "b", "m", "n", "dV", "K", "rho_a"] + (["dV_inst", "m_a"] if data.shape[1] > 7 else [])
    table = np.rec.fromarrays(data.T, names=names)
    report = json.loads(Path(prefix + ".json").read_text())
    return table, report
