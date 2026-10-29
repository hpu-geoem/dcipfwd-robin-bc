# ex01-basement-3d (Example 1 of the paper, 3D)

3D two-layer earth: 100 ohm-m cover, 50 m thick,
over a 1000 ohm-m basement with integral chargeability 0.1 (L = h rho2 / rho1 = 500 m; the homogeneous mixed condition needs
R of about 10 L, Section 2 of the paper). The same core mesh is padded to several outer extents: 500 m: 7776 cells, 1000 m: 11200 cells, 2000 m: 18304 cells, 5000 m: 28500 cells, 20000 m: 50692 cells.

Eleven surface electrodes from -100 to 100 m (20 m spacing) on the x axis record pole-pole data
from electrode 0 and dipole-dipole data with n = 1..6. Analytic responses: N-layer quadrature of scripts/analytic.py.
Runs: group example1 of scripts/paper_runs.py; background variants of the robustness checks in backgrounds/
(scripts/make_backgrounds.py).
