"""Check two properties of the trilinear LOR element matrix used in Section 3.2 of the paper.

    python3 -I scripts/checks/lor_vertex_quadrature.py

With vertex (two-point Gauss-Lobatto) quadrature, the diffusion matrix of a box couples each vertex only with its three
edge neighbours, all with negative entries, for any aspect ratio; with exact (Gauss) quadrature, stretched boxes give
positive off-diagonal entries. The generalized eigenvalues of the two matrices (on the complement of constants) lie in
[1, 3^(d-1)] = [1, 9] in 3D; for the mass matrix, vertex quadrature (lumping) changes the eigenvalues by factors in
[1, 3^d] = [1, 27].
"""

import itertools

import numpy as np

NODES = list(itertools.product([0, 1], repeat=3))


def stiffness(size, points, weights):
    def gradient(node, x):
        g = []
        for d in range(3):
            value = 1.0
            for e in range(3):
                value *= ((1.0 if node[e] else -1.0) / size[e]) if e == d else (x[e] if node[e] else 1.0 - x[e])
            g.append(value)
        return np.array(g)

    matrix = np.zeros((8, 8))
    for q in itertools.product(range(len(points)), repeat=3):
        x = [points[i] for i in q]
        w = np.prod([weights[i] for i in q]) * np.prod(size)
        g = np.array([gradient(node, x) for node in NODES])
        matrix += w * g @ g.T
    return matrix


def mass(size, points, weights):
    matrix = np.zeros((8, 8))
    for q in itertools.product(range(len(points)), repeat=3):
        x = [points[i] for i in q]
        w = np.prod([weights[i] for i in q]) * np.prod(size)
        phi = np.array([np.prod([x[e] if node[e] else 1.0 - x[e] for e in range(3)]) for node in NODES])
        matrix += w * np.outer(phi, phi)
    return matrix


def main() -> None:
    lobatto = ([0.0, 1.0], [0.5, 0.5])
    gauss = ([(1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2], [0.5, 0.5])
    for size in [(1.0, 1.0, 1.0), (10.0, 1.0, 0.2), (1.0, 50.0, 3.0)]:
        vertex, exact = stiffness(size, *lobatto), stiffness(size, *gauss)
        pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        coupled = [(i, j) for i, j in pairs if abs(vertex[i, j]) > 1e-12]
        edges = all(sum(abs(a - b) for a, b in zip(NODES[i], NODES[j])) == 1 for i, j in coupled)
        positive_exact = sum(exact[i, j] > 1e-12 for i, j in pairs)
        ratio = np.linalg.eigvals(np.linalg.pinv(exact) @ vertex).real
        ratio = np.sort(ratio[ratio > 1e-9])
        print(f"box {size}: vertex couplings {len(coupled)} (edge only: {edges}, max {max(vertex[i, j] for i, j in coupled):.3g}), "
              f"exact positive off-diagonals {positive_exact}, eigenvalue ratio [{ratio.min():.3f}, {ratio.max():.3f}]")
        lumped, consistent = mass(size, *lobatto), mass(size, *gauss)
        mass_ratio = np.sort(np.linalg.eigvals(np.linalg.solve(consistent, lumped)).real)
        print(f"  mass: lumped diagonal {np.allclose(lumped, np.diag(np.diag(lumped)))}, eigenvalue ratio "
              f"[{mass_ratio.min():.3f}, {mass_ratio.max():.3f}]")


if __name__ == "__main__":
    main()
