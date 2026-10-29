"""Prototype: a-priori error of 2.5D wavenumber rules for a two-layer earth (surface source and receivers).

V(r) = (1/pi) int_0^inf Vt(k, r) dk with Vt the full cosine transform. Two-layer image series:
Vt(k, r) = rho1/(2 pi) * 2 [K0(k r) + 2 sum_n q^n K0(k sqrt(r^2 + (2 n h)^2))].
Compares the logarithmic Gauss rule of the C++ code, weights fitted to a homogeneous half-space, and weights
fitted to the layered background, at equal numbers of wavenumbers.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import k0

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analytic  # noqa: E402


def log_rule(k_min, k_max, per_decade):
    """Mirror of make_log_wavenumber_quadrature: one node on [0, k_min] exact for a + b ln k, Gauss in ln k above."""
    x, w = np.polynomial.legendre.leggauss(per_decade)
    nodes, weights = [k_min / np.e], [k_min]  # int_0^kmin (a + b ln k) dk = kmin (a + b (ln kmin - 1))
    decades = int(np.ceil(np.log10(k_max / k_min) - 1e-12))
    edges = np.log(k_min) + np.linspace(0, np.log(k_max / k_min), decades + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        nodes += list(np.exp(t))
        weights += list(0.5 * (hi - lo) * w * np.exp(t))
    return np.array(nodes), np.array(weights)


def transformed(rho1, rho2, h, k, r, n_images):
    q = (rho2 - rho1) / (rho2 + rho1)
    n = np.arange(1, n_images + 1)
    k = np.asarray(k)[:, None, None]
    r = np.asarray(r)[None, :, None]
    terms = (q ** n)[None, None, :] * k0(k * np.sqrt(r**2 + (2 * n * h) ** 2))
    return rho1 / np.pi * (k0(k[..., 0] * r[..., 0]) + 2 * terms.sum(axis=-1))


def fit(nodes, w0, table, target, mu=1e-3):
    """Least squares for weights: sum_i w_i table[i, j] / pi = target[j], regularized toward w0 (relative)."""
    a = (table / np.pi / target[None, :]).T
    rows = np.vstack([a, mu * np.diag(1.0 / w0)])
    rhs = np.concatenate([np.ones(a.shape[0]), mu * np.ones(len(w0))])
    return np.linalg.lstsq(rows, rhs, rcond=None)[0]


def main():
    r_min, r_max = 20.0, 1000.0
    r_fit = np.geomspace(r_min, r_max, 60)
    r_test = np.geomspace(r_min, r_max, 37) * 1.013
    for rho2, n_images in ((10.0, 400), (1000.0, 40000), (0.1, 400)):
        rho1, h = 100.0, 50.0
        v_test = analytic.two_layer_surface_potential(rho1, rho2, h, r_test, n_images)
        v_fit = analytic.two_layer_surface_potential(rho1, rho2, h, r_fit, n_images)
        print(f"rho2/rho1 = {rho2 / rho1:g}")
        for per_decade in (2, 3, 4, 6):
            k_min, k_max = 1e-3 / r_max, 15.0 / r_min
            nodes, weights = log_rule(k_min, k_max, per_decade)
            lay_test = transformed(rho1, rho2, h, nodes, r_test, n_images)
            lay_fit = transformed(rho1, rho2, h, nodes, r_fit, n_images)
            hom_fit = rho1 / np.pi * k0(nodes[:, None] * r_fit[None, :])
            err = lambda w: np.max(np.abs((w @ lay_test) / np.pi / v_test - 1))
            w_hom = fit(nodes, weights, hom_fit, rho1 / (2 * np.pi * r_fit))
            w_lay = fit(nodes, weights, lay_fit, v_fit)
            print(f"  {len(nodes):2d} nodes ({per_decade}/decade): log rule {err(weights):.2e}, "
                  f"homogeneous fit {err(w_hom):.2e}, layered fit {err(w_lay):.2e}")


if __name__ == "__main__":
    main()
