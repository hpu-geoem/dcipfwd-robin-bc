"""Views of octree models from the leaf-cell files written by code/apps/octree_mesh.cc: 3D cutaway renderings
(PyVista, off screen) and slices drawn as vector polygons with the cell edges."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def read_cells(path: Path) -> np.ndarray:
    """Rows: lower[dim] upper[dim] log10_resistivity chargeability core."""
    return np.loadtxt(path)


def render_cutaway(cells: np.ndarray, column: int, cmap: str, clim, png: Path, electrodes=None,
                   depth: float = 3.0, cut=((10.0, 13.0), (7.0, 10.0), 1.6)) -> None:
    """Render the core cells above a depth (km) with a cut-out corner (x range, y range, depth) to a PNG."""
    import pyvista as pv

    core = cells[cells[:, -1] == 1]
    lower, upper = core[:, 0:3] / 1000.0, core[:, 3:6] / 1000.0
    centre = 0.5 * (lower + upper)
    (x0, x1), (y0, y1), z1 = cut
    keep = (centre[:, 2] < depth) & ~((centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0)
                                       & (centre[:, 1] < y1) & (centre[:, 2] < z1))
    core, lower, upper = core[keep], lower[keep], upper[keep]
    n = len(core)
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    points = np.empty((8 * n, 3))
    for k, (a, b, c) in enumerate(corners):
        points[k::8] = np.stack([np.where(a, upper[:, 0], lower[:, 0]), np.where(b, upper[:, 1], lower[:, 1]),
                                 -np.where(c, upper[:, 2], lower[:, 2])], axis=1)
    connectivity = np.hstack([np.full((n, 1), 8), np.arange(8 * n).reshape(n, 8)]).ravel()
    grid = pv.UnstructuredGrid(connectivity, np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8), points)
    grid.cell_data["value"] = core[:, column] * (1000.0 if column == 7 else 1.0)
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1400))
    plotter.add_mesh(surface, scalars="value", cmap=cmap, clim=clim, show_scalar_bar=False, show_edges=True,
                     edge_color="black", line_width=0.4, edge_opacity=0.35, ambient=0.35, diffuse=0.65, specular=0.0)
    if electrodes is not None:
        e = electrodes / 1000.0
        visible = ~((e[:, 0] > x0) & (e[:, 0] < x1) & (e[:, 1] > y0) & (e[:, 1] < y1))
        plotter.add_points(np.column_stack([e[visible, 0], e[visible, 1], np.full(visible.sum(), 0.01)]),
                           color="black", point_size=4, render_points_as_spheres=True)
    plotter.set_background("white")
    plotter.camera_position = [(20.5, 1.0, 6.5), (10.0, 10.0, -1.4), (0, 0, 1)]
    plotter.camera.zoom(1.25)
    plotter.screenshot(str(png), transparent_background=False)
    plotter.close()


def slice_polygons(cells: np.ndarray, axis: int, value: float, horizontal: tuple[int, int]):
    """Rectangles (in km) of the cells cut by the plane coordinate[axis] = value, and the cut cells."""
    dim = (cells.shape[1] - 3) // 2
    lower, upper = cells[:, :dim], cells[:, dim:2 * dim]
    hit = (lower[:, axis] <= value) & (upper[:, axis] > value)
    a, b = horizontal
    lo, hi = lower[hit] / 1000.0, upper[hit] / 1000.0
    polygons = np.stack([np.column_stack([lo[:, a], lo[:, b]]), np.column_stack([hi[:, a], lo[:, b]]),
                         np.column_stack([hi[:, a], hi[:, b]]), np.column_stack([lo[:, a], hi[:, b]])], axis=1)
    return polygons, cells[hit]


def draw_slice(ax, polygons, values, cmap, norm, edges: bool = True):
    collection = PolyCollection(polygons, array=values, cmap=cmap, norm=norm,
                                edgecolors=(0, 0, 0, 0.45) if edges else "face", linewidths=0.06 if edges else 0.0)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def image_axes(ax, png: Path, crop=None) -> None:
    image = plt.imread(png)
    if crop is not None:
        (r0, r1), (c0, c1) = crop
        image = image[r0:r1, c0:c1]
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()
