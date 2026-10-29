"""Reference potentials of point current electrodes for verification (unit current, z down).

- half_space_potential: homogeneous half-space with a buried or surface source (image method).
- two_layer_surface_potential: image series of a layer over a half-space (surface source and receiver).
- layered_surface_potential: N-layer earth, surface source and receiver, from the Pekeris resistivity
  transform by panel quadrature (independent of the C++ filter); also returns dV/dr.
- beta: the ratio -r dV/dr / V, which the homogeneous (Dey-Morrison) mixed condition assumes to be 1 on a
  vertical side wall.

key_201() reads the Key (2012) filter of code/include/integral_transforms.h for quick checks.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HEADER = Path(__file__).resolve().parents[1] / "code" / "include" / "integral_transforms.h"


@lru_cache(maxsize=1)
def key_201() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (base, j0 weights, j1 weights) of the Key (2012) 201-point Hankel filter."""
    text = HEADER.read_text()
    arrays = {}
    for name in ("key_201_base", "key_201_j0", "key_201_j1"):
        match = re.search(name + r"\s*=\s*\{(.*?)\};", text, re.S)
        arrays[name] = np.array([float(v) for v in re.findall(r"[-+]?\d\.\d+e[-+]\d+", match.group(1))])
        assert arrays[name].size == 201, name
    return arrays["key_201_base"], arrays["key_201_j0"], arrays["key_201_j1"]


def half_space_potential(rho: float, source, point) -> float:
    source, point = np.asarray(source, float), np.asarray(point, float)
    image = source * np.array([1.0, 1.0, -1.0])
    return rho / (4 * np.pi) * (1 / np.linalg.norm(point - source) + 1 / np.linalg.norm(point - image))


def two_layer_surface_potential(rho1: float, rho2: float, thickness: float, r, n_images: int = 4000):
    r = np.asarray(r, float)
    k = (rho2 - rho1) / (rho2 + rho1)
    n = np.arange(1, n_images + 1)
    terms = (k ** n)[None, :] / np.hypot(r[..., None], 2 * n * thickness)
    return rho1 / (2 * np.pi) * (1 / r + 2 * terms.sum(axis=-1))


def resistivity_transform(lam: np.ndarray, rho: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Pekeris recursion T_i = (T_{i+1} + rho_i tanh(lam h_i)) / (1 + T_{i+1} tanh(lam h_i) / rho_i)."""
    t = np.full_like(lam, rho[-1], dtype=float)
    for i in range(len(rho) - 2, -1, -1):
        th = np.tanh(lam * thickness[i])
        t = (t + rho[i] * th) / (1 + t * th / rho[i])
    return t


def _panel_rule(edges: np.ndarray, order: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on consecutive panels."""
    x, w = np.polynomial.legendre.leggauss(order)
    lower, upper = edges[:-1, None], edges[1:, None]
    nodes = 0.5 * (upper - lower) * x[None, :] + 0.5 * (upper + lower)
    weights = 0.5 * (upper - lower) * w[None, :]
    return nodes.ravel(), weights.ravel()


def layered_surface_potential(rho, thickness, r):
    """Return V(r) and dV/dr at the surface of an N-layer earth for a unit surface current.

    rho has N entries; thickness has N - 1 entries (the last layer is a half-space). The surface-layer term
    rho_1 / r is exact; the rest, (T(lambda) - rho_1) J0(lambda r), decays like exp(-2 lambda h_1) and is
    integrated with Gauss-Legendre panels that are logarithmic at small lambda and shorter than a quarter
    period of J0 at large lambda. Relative accuracy is about 1e-9 for contrasts up to 1e4."""
    from scipy.special import j0, j1

    rho, thickness = np.asarray(rho, float), np.asarray(thickness, float)
    r_values = np.atleast_1d(np.asarray(r, float))
    if len(rho) == 1:
        return rho[0] / (2 * np.pi * r_values), -rho[0] / (2 * np.pi * r_values**2)
    h1 = thickness[0]
    # Lowest feature: lambda ~ rho_min / (rho_max * total thickness).
    low = 1.0e-4 * rho.min() / (rho.max() * thickness.sum())
    upper = 40.0 / h1
    values, derivatives = [], []
    for radius in r_values:
        edges = np.unique(np.concatenate([[0.0], np.geomspace(low, upper, 60 * int(np.ceil(np.log10(upper / low)))),
                                          np.arange(0.0, upper, np.pi / (4.0 * radius)), [upper]]))
        lam, weight = _panel_rule(edges)
        kernel = resistivity_transform(lam, rho, thickness) - rho[0]
        values.append((rho[0] / radius + np.sum(weight * kernel * j0(lam * radius))) / (2 * np.pi))
        derivatives.append((-rho[0] / radius**2 - np.sum(weight * kernel * lam * j1(lam * radius))) / (2 * np.pi))
    return np.array(values), np.array(derivatives)


def beta(rho, thickness, r):
    """Return -r dV/dr / V at the surface; the homogeneous mixed condition assumes 1 on a vertical wall."""
    v, dv = layered_surface_potential(rho, thickness, r)
    return -np.asarray(r) * dv / v


if __name__ == "__main__":
    # Self-check against the image series.
    r = np.array([1.0, 10.0, 100.0, 1000.0])
    for ratio in (0.01, 0.1, 10.0, 1000.0):
        v, _ = layered_surface_potential([1.0, ratio], [1.0], r)
        ref = two_layer_surface_potential(1.0, ratio, 1.0, r, n_images=200000 if ratio > 100 else 4000)
        print(f"rho2/rho1={ratio:g}: max relative difference quadrature vs images {np.max(np.abs(v / ref - 1)):.2e}")
