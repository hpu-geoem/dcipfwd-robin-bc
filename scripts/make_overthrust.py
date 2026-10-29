"""Build the DC/IP examples from the SEG/EAGE Overthrust velocity model (examples 3 and 4 of the paper).

    python3 -I scripts/make_overthrust.py section    # 2.5D section, models/e8-overthrust/section2d/
    python3 -I scripts/make_overthrust.py volume     # 3D sub-volume, models/e8-overthrust/volume3d/

Source: models/e8-overthrust/source/overthrust_3D_true_model.h5 (Zenodo 10.5281/zenodo.4252588, CC-BY 4.0), squared
slowness m = 1/v^2 in s^2/km^2 on an 801 x 801 x 207 grid with 25 m spacing, stored as m[z, y, x]; the top 20 cells are
a 1.5 km/s water layer added to the original 187-cell model and are dropped here, so z = 0 is the top of the rock.

Rules defined for this study (state them in the paper):
- resistivity rho = 10 (v / 2.18)^6 ohm-m, a Faust-type power law (2.18 km/s is the slowest rock velocity);
- integral chargeability eta = 0.02 + 0.13 exp(-((v - 3.0) / 0.25)^2);
- coarse cells take the mean of log10(rho) and of eta over the 25 m cells they contain;
- the selected part of the model is embedded in a horizontally layered background whose layers are the lateral log
  means of the selected part over fixed depth intervals; padding cells outside the selected part take these values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcipio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "e8-overthrust"
SOURCE = BASE / "source" / "overthrust_3D_true_model.h5"
DX = 25.0
WATER_CELLS = 20
BACKGROUND_TOPS = [0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]


def rho_of(v):
    return 10.0 * (v / 2.18) ** 6


def eta_of(v):
    return 0.02 + 0.13 * np.exp(-(((v - 3.0) / 0.25) ** 2))


def load(x_range, y_range):
    """Return log10(rho) and eta on the 25 m grid, indexed [z, y, x], for x and y ranges in metres."""
    import h5py

    ix = slice(int(round(x_range[0] / DX)), int(round(x_range[1] / DX)))
    iy = slice(int(round(y_range[0] / DX)), int(round(y_range[1] / DX)))
    with h5py.File(SOURCE, "r") as f:
        m = np.asarray(f["m"][WATER_CELLS:, iy, ix], dtype=float)
    v = 1.0 / np.sqrt(m)
    return np.log10(rho_of(v)), eta_of(v)


def coarse_axis(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return start + step * np.arange(count + 1)


def depth_nodes() -> np.ndarray:
    """Vertical nodes of the selected part: 25 m to 500 m, 50 m to 1500 m, 125 m to 4625 m and 50 m to 4675 m."""
    nodes = np.concatenate([coarse_axis(0.0, 500.0, 25.0), coarse_axis(500.0, 1500.0, 50.0)[1:],
                            coarse_axis(1500.0, 4625.0, 125.0)[1:], [4675.0]])
    assert np.all(np.diff(nodes) > 0) and nodes[-1] == 187 * DX
    return nodes


def block_mean(fine: np.ndarray, fine_coordinates: list[np.ndarray], coarse_nodes: list[np.ndarray]) -> np.ndarray:
    """Average fine cell values (centres given per axis) into the coarse cells between the coarse nodes."""
    result = fine
    for axis, (centres, nodes) in enumerate(zip(fine_coordinates, coarse_nodes)):
        assert nodes[0] <= centres.min() and centres.max() <= nodes[-1], "fine cells outside the coarse grid"
        index = np.searchsorted(nodes, centres, side="right") - 1
        sums = np.zeros(result.shape[:axis] + (len(nodes) - 1,) + result.shape[axis + 1:])
        counts = np.zeros(len(nodes) - 1)
        np.add.at(counts, index, 1.0)
        moved = np.moveaxis(result, axis, 0)
        target = np.moveaxis(sums, axis, 0)
        np.add.at(target, index, moved)
        result = np.moveaxis(target, 0, axis) / counts.reshape([-1 if a == axis else 1 for a in range(result.ndim)])
    return result


def background(log_rho: np.ndarray, eta: np.ndarray, z_nodes: np.ndarray) -> list[tuple[float, float, float]]:
    """Layered background: lateral means of log10(rho) and eta of the coarse cells within the fixed depth intervals."""
    centres = 0.5 * (z_nodes[1:] + z_nodes[:-1])
    tops = BACKGROUND_TOPS + [np.inf]
    layers = []
    for top, bottom in zip(tops[:-1], tops[1:]):
        rows = (centres >= top) & (centres < bottom)
        layers.append((top, float(10.0 ** log_rho[rows].mean()), float(eta[rows].mean())))
    return layers


def background_value(layers, z):
    value = layers[0]
    for layer in layers:
        if z >= layer[0]:
            value = layer
    return value


def padded(nodes: np.ndarray, extent: float, growth: float = 1.3, lower: bool = True, upper: bool = True) -> np.ndarray:
    """Extend node coordinates by cells growing from the end cells until the axis spans extent beyond each end."""
    result = list(nodes)
    if upper:
        cell = nodes[-1] - nodes[-2]
        while result[-1] < nodes[-1] + extent:
            cell *= growth
            result.append(result[-1] + cell)
    if lower:
        cell = nodes[1] - nodes[0]
        while result[0] > nodes[0] - extent:
            cell *= growth
            result.insert(0, result[0] - cell)
    return np.array(result)


def write_model(directory: Path, name: str, axes: list[np.ndarray], core_axes: list[np.ndarray], log_rho, eta, layers):
    """Write a padded tensor mesh: cells inside the core take the coarse values, others the layered background."""
    mesh = dcipio.TensorMesh(axes)
    centres = [0.5 * (a[1:] + a[:-1]) for a in axes]
    inside = []
    index = []
    for axis, (c, core) in enumerate(zip(centres, core_axes)):
        inside.append((c > core[0]) & (c < core[-1]))
        index.append(np.clip(np.searchsorted(core, c) - 1, 0, len(core) - 2))
    grids = np.meshgrid(*index, indexing="ij")
    masks = np.meshgrid(*inside, indexing="ij")
    in_core = np.logical_and.reduce(masks)
    # Coarse arrays are indexed [z, y, x] (3D) or [z, y] (2D); mesh axes are (x, y, z) or (y, z).
    core_values_rho = log_rho[tuple(grids[::-1])]
    core_values_eta = eta[tuple(grids[::-1])]
    depth = np.meshgrid(*centres, indexing="ij")[-1]
    bg_rho = np.vectorize(lambda z: np.log10(background_value(layers, z)[1]))(depth)
    bg_eta = np.vectorize(lambda z: background_value(layers, z)[2])(depth)
    cell_rho = np.where(in_core, core_values_rho, bg_rho)
    cell_eta = np.where(in_core, core_values_eta, bg_eta)
    mesh.attributes = np.arange(1, cell_rho.size + 1).reshape(cell_rho.shape)
    mesh.write_mfem(directory / f"{name}.mesh")
    properties = {int(a): (float(10.0 ** r), float(e)) for a, r, e in zip(mesh.attributes.ravel(), cell_rho.ravel(), cell_eta.ravel())}
    dcipio.write_properties(directory / f"{name}.prop", properties)
    return mesh.shape


def survey(positions: np.ndarray, along: str, offset: float = 0.0, max_n: int = 10) -> dcipio.Survey:
    """Dipole-dipole and pole-dipole (pole at infinity) rows on one line, n = 1..max_n."""
    electrodes = dcipio.line_electrodes(positions, along=along)
    if along == "x":
        electrodes[:, 1] = offset
    n = len(positions)
    rows = list(dcipio.dipole_dipole_rows(n, max_n))
    for k in range(1, max_n + 1):
        for i in range(n - 1 - k):
            rows.append([i, -1, i + k, i + k + 1])
    return dcipio.Survey(electrodes, np.array(rows, dtype=int))


def section(extents):
    """2.5D: section y = 10 km, x from 7 to 13 km (model y), embedded in its layered background; line from 8 to 12 km."""
    directory = BASE / "section2d"
    directory.mkdir(parents=True, exist_ok=True)
    x_range, y_plane = (7000.0, 13000.0), 10000.0
    log_rho, eta = load(x_range, (y_plane, y_plane + DX))
    log_rho, eta = log_rho[:, 0, :], eta[:, 0, :]  # [z, x]
    fine = [DX * (np.arange(log_rho.shape[0]) + 0.5), x_range[0] + DX * (np.arange(log_rho.shape[1]) + 0.5)]
    z_nodes = depth_nodes()
    y_nodes = coarse_axis(*x_range, 50.0)
    log_rho = block_mean(log_rho, fine, [z_nodes, y_nodes])
    eta = block_mean(eta, fine, [z_nodes, y_nodes])
    layers = background(log_rho, eta, z_nodes)
    dcipio.write_background(directory / "background.txt", layers)
    sizes = {}
    for extent in extents:
        axes = [padded(y_nodes, extent), padded(z_nodes, extent, lower=False)]
        sizes[extent] = write_model(directory, f"model-{int(extent)}", axes, [y_nodes, z_nodes], log_rho, eta, layers)
    survey(np.arange(8000.0, 12000.0 + 1e-9, 100.0), "y").write(directory / "survey.txt")
    np.savez_compressed(directory / "core.npz", y=y_nodes, z=z_nodes, log_rho=log_rho, eta=eta)
    (directory / "sizes.json").write_text(json.dumps({str(int(k)): v for k, v in sizes.items()}))
    print("section2d", {int(k): int(np.prod(v)) for k, v in sizes.items()}, "background", layers)


def volume(extents):
    """3D: x from 7.5 to 12.5 km, y from 9 to 11 km, embedded in its layered background; three lines x 8-12 km."""
    directory = BASE / "volume3d"
    directory.mkdir(parents=True, exist_ok=True)
    x_range, y_range = (7500.0, 12500.0), (9000.0, 11000.0)
    log_rho, eta = load(x_range, y_range)  # [z, y, x]
    fine = [DX * (np.arange(log_rho.shape[0]) + 0.5), y_range[0] + DX * (np.arange(log_rho.shape[1]) + 0.5),
            x_range[0] + DX * (np.arange(log_rho.shape[2]) + 0.5)]
    z_nodes = depth_nodes()
    x_nodes, y_nodes = coarse_axis(*x_range, 100.0), coarse_axis(*y_range, 100.0)
    log_rho = block_mean(log_rho, fine, [z_nodes, y_nodes, x_nodes])
    eta = block_mean(eta, fine, [z_nodes, y_nodes, x_nodes])
    layers = background(log_rho, eta, z_nodes)
    dcipio.write_background(directory / "background.txt", layers)
    sizes = {}
    for extent in extents:
        axes = [padded(x_nodes, extent), padded(y_nodes, extent), padded(z_nodes, extent, lower=False)]
        sizes[extent] = write_model(directory, f"model-{int(extent)}", axes, [x_nodes, y_nodes, z_nodes], log_rho, eta, layers)
        # Spec for scripts/mesh_from_spec.py on machines without numpy.
        spec = {"name": f"model-{int(extent)}", "axes": [a.tolist() for a in axes],
                "core_axes": [x_nodes.tolist(), y_nodes.tolist(), z_nodes.tolist()], "core_shape": list(log_rho.shape),
                "core_log_rho": log_rho.ravel().tolist(), "core_eta": eta.ravel().tolist(), "background": layers}
        (directory / f"spec-{int(extent)}.json").write_text(json.dumps(spec))
    positions = np.arange(8000.0, 12000.0 + 1e-9, 100.0)
    lines = [survey(positions, "x", offset) for offset in (9500.0, 10000.0, 10500.0)]
    electrodes = np.concatenate([line.electrodes for line in lines])
    rows = np.concatenate([np.where(line.rows >= 0, line.rows + i * len(positions), -1) for i, line in enumerate(lines)])
    dcipio.Survey(electrodes, rows).write(directory / "survey.txt")
    np.savez_compressed(directory / "core.npz", x=x_nodes, y=y_nodes, z=z_nodes, log_rho=log_rho, eta=eta)
    (directory / "sizes.json").write_text(json.dumps({str(int(k)): v for k, v in sizes.items()}))
    print("volume3d", {int(k): int(np.prod(v)) for k, v in sizes.items()}, "background", layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["section", "volume"])
    parser.add_argument("--extents", type=float, nargs="*", default=[250.0, 500.0, 1000.0, 3000.0, 10000.0, 200000.0])
    args = parser.parse_args()
    (section if args.part == "section" else volume)(args.extents)
