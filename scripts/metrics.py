"""Error, cost and diagnostic measures of dcipfwd runs, shared by the tables and figures of the paper.

Errors of a run against a reference on the same core mesh (truncation error) or against an analytic solution (total
error): relative errors of the DC response dV (= of rho_a) and of the instantaneous response dV_inst, and absolute
errors of the apparent chargeability m_a = 1 - dV_inst / dV in mV/V. With eps_dc and eps_inst the relative errors of
the two states and m the reference chargeability, the chargeability error is exactly
    m_a - m = (1 - m) (eps_dc - eps_inst) / (1 + eps_dc),
so it depends on the difference of the two state errors, not on their size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcipio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read(path: str | Path):
    """Return the response table and run report of an output prefix (relative to the project root or absolute)."""
    path = Path(path)
    return dcipio.read_result(path if path.is_absolute() else ROOT / path)


def check_rows(table, reference, name: str = "") -> None:
    if len(table) != len(reference):
        raise ValueError(f"{name}: {len(table)} rows, reference {len(reference)}")
    for field in ("a", "b", "m", "n"):
        if not np.array_equal(table[field], reference[field]):
            raise ValueError(f"{name}: measurement index '{field}' differs from the reference")


def masks(table) -> dict[str, np.ndarray]:
    """Rows of all data, of data with a remote current electrode (pole) and of data with two current electrodes."""
    return {"all": np.ones(len(table), bool), "pole": table.b < 0, "dipole": table.b >= 0}


def summary(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def state_errors(table, reference) -> dict[str, np.ndarray]:
    """Per-datum relative errors of the two states and absolute chargeability errors (mV/V)."""
    result = {"dc": table.dV / reference.dV - 1.0}
    if "m_a" in table.dtype.names and "m_a" in reference.dtype.names:
        result["inst"] = table.dV_inst / reference.dV_inst - 1.0
        result["ma"] = (table.m_a - reference.m_a) * 1000.0
    return result


def errors(table, reference, name: str = "") -> dict:
    """Median, 99th percentile and largest error of rho_a (relative) and m_a (mV/V) over the row groups."""
    check_rows(table, reference, name)
    per_datum = state_errors(table, reference)
    out = {}
    for group, mask in masks(table).items():
        if not mask.any():
            continue
        out[group] = {"rho": summary(np.abs(per_datum["dc"][mask]))}
        if "ma" in per_datum:
            out[group]["inst"] = summary(np.abs(per_datum["inst"][mask]))
            out[group]["ma"] = summary(np.abs(per_datum["ma"][mask]))
    return out


def ip_propagation(table, reference) -> dict:
    """How the errors of the DC and instantaneous states combine in m_a: correlation of the two state errors, the
    size of their difference relative to the DC error, and the check of the exact propagation formula."""
    e = state_errors(table, reference)
    if "ma" not in e:
        return {}
    dc, inst = e["dc"], e["inst"]
    predicted = (1.0 - reference.m_a) * (dc - inst) / (1.0 + dc) * 1000.0
    small = np.abs(dc) > 1e-14
    defined = small.any() and np.std(dc) > 0.0 and np.std(inst) > 0.0
    return {"correlation": float(np.corrcoef(dc, inst)[0, 1]) if defined else None,
            "difference_over_dc_median": float(np.median(np.abs(dc - inst)[small] / np.abs(dc)[small])) if small.any() else None,
            "dc": summary(np.abs(dc)), "inst": summary(np.abs(inst)), "difference": summary(np.abs(dc - inst)),
            "formula_residual_mvv": float(np.abs(predicted - e["ma"]).max())}


def cost(report: dict) -> dict:
    """Sizes, times (seconds, summed over the DC and instantaneous passes) and memory of a run."""
    solvers = [s for p in report["passes"] for s in p["solvers"]]
    first = solvers[0]
    out = {"ranks": report["ranks"], "order": report["order"], "preconditioner": report["preconditioner"],
           "solver": report["solver"], "dofs": first["dofs"], "cells": first["cells"],
           "wavenumbers": first["wavenumbers"], "passes": len(report["passes"]),
           "wall": float(sum(p["wall_time"] for p in report["passes"])),
           "wall_dc": float(report["passes"][0]["wall_time"]),
           "refine": float(sum(s["refine_time"] for s in solvers)),
           "fe": float(sum(s["fe_time"] for s in solvers)),
           "boundary": float(sum(s["boundary_time"] for s in solvers)),
           "assemble": float(sum(s["assemble_time"] for s in solvers)),
           "preconditioner_time": float(sum(s["preconditioner_time"] for s in solvers)),
           "solve": float(sum(s["solve_time"] for s in solvers)),
           "electrode_boundary": float(sum(s["electrode_boundary_time"] for s in solvers)),
           "max_iterations": int(max(s["max_iterations"] for s in solvers)),
           "total_iterations": int(sum(s["total_iterations"] for s in solvers)),
           "solves": int(sum(s["n_solves"] for s in solvers)),
           "memory_gib_per_rank": float(report["peak_memory_mib_max"]) / 1024.0,
           "memory_gib_total": float(report["peak_memory_mib_sum"]) / 1024.0}
    out["ms_per_iteration"] = 1000.0 * out["solve"] / out["total_iterations"] if out["total_iterations"] else None
    for key in ("boundary_points", "clipped_points", "electrode_points", "electrode_clipped_points", "unbounded_points"):
        out[key] = int(sum(s.get(key, 0) for s in solvers))
    ratios = [(s["ratio_min"], s["ratio_max"]) for s in solvers if s.get("ratio_min") is not None]
    out["ratio_min"] = min(r[0] for r in ratios) if ratios else None
    out["ratio_max"] = max(r[1] for r in ratios) if ratios else None
    if "rule_dc" in report:
        out["rule_error_dc"] = report["rule_dc"]["error"]
        out["rule_error_inst"] = report["rule_instantaneous"]["error"]
    return out


def record(path: str | Path, reference: str | Path | None) -> dict:
    """Errors against a reference (if given), IP propagation and cost of one run."""
    table, report = read(path)
    out = {"source": str(path), "cost": cost(report)}
    if reference is not None:
        ref, _ = read(reference)
        out["errors"] = errors(table, ref, str(path))
        out["ip"] = ip_propagation(table, ref)
    return out


def reciprocity(path: str | Path) -> dict | None:
    """Relative differences |V_ij - V_ji| / |V_ij| of the unit-current potentials of electrode pairs that are both
    current and potential electrodes (<prefix>.pot), for the DC and the instantaneous state."""
    path = Path(path)
    pot = (path if path.is_absolute() else ROOT / path).with_suffix(".pot")
    if not pot.exists():
        return None
    data = np.loadtxt(pot, comments="#", ndmin=2)
    index = {(int(a), int(m)): i for i, (a, m) in enumerate(data[:, :2])}
    pairs = [(i, index[(m, a)]) for (a, m), i in index.items() if a < m and (m, a) in index]
    if not pairs:
        return None
    first, second = np.array(pairs).T
    out = {"pairs": len(pairs)}
    for column, state in ((2, "dc"), (3, "inst")):
        if data.shape[1] > column:
            diff = np.abs(data[first, column] - data[second, column]) / np.abs(data[first, column])
            out[state] = summary(diff)
    return out


def require(paths: list[str | Path]) -> None:
    """Stop with a list of the missing run reports among the given output prefixes."""
    missing = [str(p) for p in paths if not ((Path(p) if Path(p).is_absolute() else ROOT / p).parent
                                            / f"{Path(p).name}.json").exists()]
    if missing:
        raise SystemExit("missing run reports:\n  " + "\n  ".join(missing))


def cheapest(records: dict[str, dict], keys: list[str], error, threshold: float) -> str | None:
    """Return the key of the run with the smallest wall time among those whose error is at most the threshold."""
    passing = [k for k in keys if k in records and error(records[k]) <= threshold]
    return min(passing, key=lambda k: records[k]["cost"]["wall"]) if passing else None
