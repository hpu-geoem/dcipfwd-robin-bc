"""Mismatch of the homogeneous (Dey-Morrison) mixed condition over a two-layer earth.

On a vertical side wall at distance R from a surface source, the homogeneous condition imposes
-dV/dr / V = 1 / R, while the layered potential has -dV/dr / V = beta(R) / R. The script tabulates beta at the
surface against R / h for several contrasts and writes figures/data/beta-two-layer.csv for the paper figure.

    python3 -I scripts/beta_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analytic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RATIOS = [1.0e-2, 1.0e-1, 1.0e1, 1.0e2, 1.0e3]


def main() -> None:
    r_over_h = np.geomspace(1.0, 1.0e4, 81)
    columns = [r_over_h]
    print("rho2/rho1 | beta at R/h = 1, 10, 100, 1000, 10000 | R/h where |beta - 1| < 0.02")
    for ratio in RATIOS:
        b = analytic.beta([1.0, ratio], [1.0], r_over_h)
        columns.append(b)
        ok = np.nonzero(np.abs(b - 1.0) >= 0.02)[0]
        reach = r_over_h[ok[-1] + 1] if ok.size and ok[-1] + 1 < r_over_h.size else (r_over_h[0] if not ok.size else np.inf)
        picks = [b[np.argmin(np.abs(r_over_h - x))] for x in (1, 10, 100, 1000, 10000)]
        print(f"{ratio:9g} | " + " ".join(f"{p:6.3f}" for p in picks) + f" | {reach:.3g}")
    out = ROOT / "figures" / "data" / "beta-two-layer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "R_over_h," + ",".join(f"beta_ratio_{r:g}" for r in RATIOS)
    np.savetxt(out, np.column_stack(columns), delimiter=",", header=header, comments="")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
