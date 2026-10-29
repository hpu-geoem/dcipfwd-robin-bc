"""Pseudosection grids of dipole-dipole and pole-dipole data on one line.

Data are plotted at the midpoint between the centre of the current electrodes and the centre of the potential dipole,
at the pseudodepth where 45-degree lines from the two centres meet: (n + 1) a / 2 for dipole-dipole and (n + 1/2) a / 2
for pole-dipole data, with the electrode spacing a and the separation n.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize


def line_coordinate(electrodes: np.ndarray) -> np.ndarray:
    """Return the coordinate along the line: x in 3D surveys, y in 2.5D surveys (the one that varies)."""
    spread = electrodes.max(axis=0) - electrodes.min(axis=0)
    return electrodes[:, int(np.argmax(spread[:2]))]


def pseudodepth(level, spacing: float, kind: str):
    """Pseudodepth of separation level n for electrode spacing a."""
    return (np.asarray(level) + (1.0 if kind == "dd" else 0.5)) * spacing / 2.0


def grid(table, electrodes: np.ndarray, values: np.ndarray, kind: str, line=None):
    """Return (midpoints, levels, array[level, midpoint]) of one array type; kind is 'dd' or 'pd'.

    The level is the separation n between the inner current and potential electrodes. line selects the electrodes of
    one line (index array) in multi-line surveys."""
    x = line_coordinate(electrodes)
    pole = table.b < 0
    rows = ~pole if kind == "dd" else pole
    if line is not None:
        rows &= np.isin(table.a, line)
    a, b, m, n = (table[k][rows].astype(int) for k in ("a", "b", "m", "n"))
    v = values[rows]
    if kind == "dd":
        mid = 0.25 * (x[a] + x[b] + x[m] + x[n])
        level = m - b
    else:
        mid = 0.5 * (x[a] + 0.5 * (x[m] + x[n]))
        level = m - a
    mids = np.unique(np.round(mid, 6))
    levels = np.arange(1, level.max() + 1)
    out = np.full((len(levels), len(mids)), np.nan)
    out[level - 1, np.searchsorted(mids, np.round(mid, 6))] = v
    return mids, levels, out


def edges(centres: np.ndarray) -> np.ndarray:
    """Return cell edges around equally or unequally spaced centres."""
    half = np.diff(centres) / 2.0
    return np.concatenate([[centres[0] - half[0]], centres[:-1] + half, [centres[-1] + half[-1]]])


def draw(ax, mids: np.ndarray, depths: np.ndarray, array: np.ndarray, cmap, norm=None):
    """Draw a pseudosection row by row, each row centred at its pseudodepth and with cells around its own midpoints
    (dipole-dipole midpoints of odd and even levels are offset by half an electrode spacing). Return the last mesh."""
    finite = array[np.isfinite(array)]
    norm = norm if norm is not None else Normalize()
    if norm.vmin is None:
        norm.vmin = finite.min()
    if norm.vmax is None:
        norm.vmax = finite.max()
    half = 0.5 * np.min(np.diff(depths)) if len(depths) > 1 else 0.5
    mesh = None
    for row, depth in enumerate(depths):
        present = np.isfinite(array[row])
        if present.sum() < 2:
            continue
        mesh = ax.pcolormesh(edges(mids[present]), [depth - half, depth + half], array[row, present][None, :], cmap=cmap,
                             norm=norm, shading="flat")
    return mesh
