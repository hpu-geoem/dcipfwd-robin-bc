"""Metrics file and LaTeX tables of the paper from the run reports.

    python3 -I scripts/figures/paper_tables.py

Reads results/example{1,2,3,4}* and results/robustness (written by scripts/paper_runs.py), checks that every response
file lists the measurements of its reference in the same order, and writes figures/data/paper-metrics.json and
manuscript/tables/*.tex (supp-*.tex: tables of the supplementary material). Truncation errors are differences from a
reference on the same core mesh with a far boundary; total errors (example 1 only) are differences from the analytic
two-layer responses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analytic  # noqa: E402
import metrics  # noqa: E402
import paper_runs  # noqa: E402
import style  # noqa: E402

ROOT = style.ROOT
TABLES = ROOT / "manuscript" / "tables"
METRICS = ROOT / "figures" / "data" / "paper-metrics.json"
COMPARED = paper_runs.COMPARED
LABELS = {"ds": "DS", "de": "DE", "ls": "LS", "le": "LE", "li": "LI", "lm": "LM"}
EXTENTS1 = (500, 1000, 2000, 5000)
EXTENTS = (500, 1000, 2000, 5000, 10000)


def sci(value: float, digits: int = 1) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def mvv(value: float) -> str:
    """Chargeability error in mV/V: fixed point down to 0.001, scientific below."""
    return f"{value:.2f}" if value >= 1.0 else f"{value:.3f}" if value >= 1e-3 else sci(value)


def write(name: str, lines: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text("\n".join(lines) + "\n")


def collect(names: dict[str, str], reference: str) -> dict:
    metrics.require(list(names.values()) + [reference])
    return {key: metrics.record(path, reference) | {"reciprocity": metrics.reciprocity(path)}
            for key, path in names.items()}


# Example 1 -----------------------------------------------------------------------------------------------------

RHO, THICKNESS, ETA = [100.0, 1000.0], [50.0], 0.1


def analytic_table(table, x: np.ndarray):
    """Analytic dV and dV_inst of the rows of a response table on the two-layer earth (electrodes 20 m apart)."""
    offsets = np.arange(20.0, 201.0, 20.0)
    dc, _ = analytic.layered_surface_potential(RHO, THICKNESS, offsets)
    inst, _ = analytic.layered_surface_potential([RHO[0], RHO[1] * (1.0 - ETA)], THICKNESS, offsets)

    def difference(values):
        def potential(i, j):
            return 0.0 if i < 0 or j < 0 else values[int(round(abs(x[i] - x[j]) / 20.0)) - 1]
        return np.array([potential(a, m) - potential(a, n) - potential(b, m) + potential(b, n)
                         for a, b, m, n in zip(*(table[k].astype(int) for k in ("a", "b", "m", "n")))])

    reference = table.copy()
    reference.dV = difference(dc)
    reference.dV_inst = difference(inst)
    reference.rho_a = reference.dV * table.K
    reference.m_a = 1.0 - reference.dV_inst / reference.dV
    return reference


def example1(out: dict) -> None:
    x = paper_runs_survey_x()
    for case in ("example1-3d", "example1-25d"):
        reference = f"results/{case}/le-20000"
        names = {f"{m}-{e}": f"results/{case}/{m}-{e}" for m in COMPARED + ("lm",) for e in EXTENTS1 + (20000,)}
        records = collect(names, reference)
        for key, path in names.items():
            if key in records:
                table, _ = metrics.read(path)
                records[key]["total"] = metrics.errors(table, analytic_table(table, x), path)
        # Background perturbations and decay switch, against the reference with the true background.
        robust = {}
        for extent in (500, 1000):
            for background in paper_runs.EXAMPLE1_BACKGROUNDS:
                for method in ("ls", "le", "li"):
                    key = f"{method}-{background}-{extent}"
                    path = f"results/robustness/{case}/{key}"
                    metrics.require([path])
                    robust[key] = metrics.record(path, reference)
        for switch in (10, 20, 25):
            for method in ("ls", "le", "li"):
                key = f"{method}-kl{switch}-500"
                path = f"results/robustness/{case}/{key}"
                if case == "example1-25d":
                    metrics.require([path])
                    robust[key] = metrics.record(path, reference)
                    robust[key]["against_kl15"] = metrics.record(path, f"results/{case}/{method}-500")["errors"]
        out[case] = {"runs": records, "robustness": robust}
    # Paper: most compact domain; supplementary material: 0.5 and 1 km.
    for name, extents in (("example1-errors.tex", (500,)), ("supp-example1-errors.tex", (500, 1000))):
        lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{rlrrrrrrr}", r"\toprule",
                 r" & & \multicolumn{3}{c}{3-D} & \multicolumn{3}{c}{2.5-D} & \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r"$R$ (km) & & PP & DD & $m_a$ (mV/V) & PP & DD & $m_a$ (mV/V) & Wall (s) \\", r"\midrule"]
        for extent in extents:
            for method in COMPARED:
                cells = []
                for case in ("example1-3d", "example1-25d"):
                    e = out[case]["runs"][f"{method}-{extent}"]["errors"]
                    cells += [sci(e["pole"]["rho"]["max"]), sci(e["dipole"]["rho"]["max"]),
                              mvv(max(e["pole"]["ma"]["max"], e["dipole"]["ma"]["max"]))]
                wall = out["example1-3d"]["runs"][f"{method}-{extent}"]["cost"]["wall"]
                lines.append(f"{extent / 1000:g} & {LABELS[method]} & " + " & ".join(cells) + f" & {wall:.1f} " + r"\\")
            lines.append(r"\addlinespace[2pt]")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        write(name, lines)


def paper_runs_survey_x() -> np.ndarray:
    rows = [ln.split("#")[0].strip() for ln in (ROOT / "models" / "ex01-basement-3d" / "survey.txt").read_text().splitlines()]
    rows = [ln for ln in rows if ln]
    n = int(rows[0])
    return np.array([[float(v) for v in ln.split()] for ln in rows[1:1 + n]])[:, 0]


# Example 2 -----------------------------------------------------------------------------------------------------

EXAMPLE2 = {"ds-amg": "DS, AMG", "de-amg": "DE, AMG", "ls-amg": "LS, AMG", "ls-mumps": "LS, MUMPS",
            "le-amg": "LE, AMG", "le-mumps": "LE, MUMPS PC", "li-amg": "LI, AMG", "li-mumps": "LI, MUMPS",
            "lm-amg": "LM, AMG", "lm-mumps": "LM, MUMPS"}


def example2(out: dict) -> None:
    for case in ("example2-3d", "example2-25d"):
        names = {f"{v}-{e}": f"results/{case}/{v}-{e}" for v in EXAMPLE2 for e in (1000, 2000)}
        out[case] = {"runs": collect(names, f"results/{case}/le-amg-20000")}
        le = {e: metrics.read(f"results/{case}/le-amg-{e}")[0] for e in (1000, 2000)}
        for key, path in names.items():
            extent = int(key.rsplit("-", 1)[1])
            table, _ = metrics.read(path)
            out[case]["runs"][key]["against_le"] = metrics.errors(table, le[extent], path)
    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Boundary, solver & $R$ (km) & PP max. & DD max. & CG its & Wall (s) & GiB/rank & Recipr. \\", r"\midrule"]
    for variant, label in EXAMPLE2.items():
        for extent in (1000, 2000):
            m = out["example2-3d"]["runs"][f"{variant}-{extent}"]
            its = "--" if m["cost"]["max_iterations"] == 0 else str(m["cost"]["max_iterations"])
            reciprocity = m["reciprocity"]["dc"]["max"] if m["reciprocity"] else float("nan")
            lines.append(f"{label if extent == 1000 else ''} & {extent / 1000:g} & {sci(m['errors']['pole']['rho']['max'])} & "
                         f"{sci(m['errors']['dipole']['rho']['max'])} & {its} & {m['cost']['wall']:.1f} & "
                         f"{m['cost']['memory_gib_per_rank']:.2f} & {sci(reciprocity)} " + r"\\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("example2-3d.tex", lines)


# Example 3 -----------------------------------------------------------------------------------------------------

def example3(out: dict) -> None:
    base = "results/example3"
    reference = f"{base}/le-dense-200000"
    names = {"le-dense-400000": f"{base}/le-dense-400000", "le-dense-p3-200000": f"{base}/le-dense-p3-200000",
             "le-layered10-200000": f"{base}/le-layered10-200000", "le-log10-200000": f"{base}/le-log10-200000",
             "le-dense-200000": reference}
    names |= {f"{m}-dense-{e}": f"{base}/{m}-dense-{e}" for m in COMPARED for e in EXTENTS}
    names |= {f"le-{r}{p}-200000": f"{base}/le-{r}{p}-200000"
              for r in ("log", "layered", "layered-joint", "layered-separate") for p in (2, 3, 4)}
    names |= {f"{m}-{r}4-{e}": f"{base}/{m}-{r}4-{e}" for m in COMPARED for r in ("log", "layered") for e in EXTENTS}
    records = collect(names, reference)
    robust = {}
    for background in ("true",) + paper_runs.EXAMPLE3_BACKGROUNDS:
        for method in ("ls", "le", "li"):
            key = f"{method}-{background}-1000"
            path = f"results/robustness/example3/{key}"
            metrics.require([path])
            robust[key] = metrics.record(path, reference)
    for switch in (10, 20, 25):
        for method in ("ls", "le", "li"):
            key = f"{method}-kl{switch}-1000"
            path = f"results/robustness/example3/{key}"
            metrics.require([path])
            robust[key] = metrics.record(path, reference)
            robust[key]["against_kl15"] = metrics.record(path, f"{base}/{method}-dense-1000")["errors"]
    out["example3"] = {"runs": records, "robustness": robust}

    rules = {"log": "Gauss", "layered": "DC fit", "layered-joint": "Joint fit", "layered-separate": "Separate fits"}
    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Weights & $N_k$ & \multicolumn{2}{c}{$\rho_a$ error} & \multicolumn{2}{c}{$m_a$ error (mV/V)} & "
             r"\multicolumn{2}{c}{$\epsilon_{\rm bg}$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r" & & Median & 99\% & Median & 99\% & DC & Inst. \\", r"\midrule"]
    for ppd in (2, 3, 4):
        for rule, label in rules.items():
            m = records[f"le-{rule}{ppd}-200000"]
            e, c = m["errors"]["all"], m["cost"]
            lines.append(f"{label} & {c['wavenumbers']} & {sci(e['rho']['median'])} & {sci(e['rho']['p99'])} & "
                         f"{e['ma']['median']:.3f} & {e['ma']['p99']:.3f} & {sci(c['rule_error_dc'])} & "
                         f"{sci(c['rule_error_inst'])} " + r"\\")
        if ppd != 4:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("example3-quadrature.tex", lines)


# Example 4 -----------------------------------------------------------------------------------------------------

THRESHOLDS = {"rho_p99": (1e-2, 3e-3, 1e-3), "ma_p99": (1.0, 0.3, 0.1)}


def example4(out: dict) -> None:
    base = "results/example4"
    reference = f"{base}/le-200000"
    names = {f"{m}-{e}": f"{base}/{m}-{e}" for m in COMPARED for e in EXTENTS}
    names |= {"le-400000": f"{base}/le-400000"}
    names |= {f"repeat-{m}-{e}": f"{base}/repeat-{m}-{e}" for m, e in (("ls", 1000), ("le", 1000), ("li", 1000), ("ds", 5000))}
    records = collect(names, reference)
    for order in (2, 3):
        for pc in ("lor", "amg"):
            key = f"central-le-p{order}-{pc}-1000"
            metrics.require([f"{base}/{key}"])
            records[key] = metrics.record(f"{base}/{key}", None)
    p2, _ = metrics.read(f"{base}/central-le-p2-lor-1000")
    p3, _ = metrics.read(f"{base}/central-le-p3-lor-1000")
    records["central-p3-vs-p2"] = metrics.errors(p2, p3)
    # Cheapest run of each method that reaches each accuracy threshold.
    fixed = {}
    for measure, thresholds in THRESHOLDS.items():
        get = (lambda r: r["errors"]["all"]["rho"]["p99"]) if measure == "rho_p99" else (lambda r: r["errors"]["all"]["ma"]["p99"])
        for threshold in thresholds:
            for method in COMPARED:
                key = metrics.cheapest(records, [f"{method}-{e}" for e in EXTENTS], get, threshold)
                fixed[f"{measure}-{threshold:g}-{method}"] = key
    out["example4"] = {"runs": records, "fixed_accuracy": fixed}

    # Paper: a compact and a large domain; supplementary material: all extents and the time per CG iteration.
    for name, extents, per_iteration in (("example4-accuracy.tex", (1000, 10000), False),
                                         ("supp-example4-accuracy.tex", EXTENTS, True)):
        lines = [r"\small", r"\setlength{\tabcolsep}{2.6pt}",
                 r"\begin{tabular}{rlrrrrrrr" + ("r" if per_iteration else "") + "}", r"\toprule",
                 r"$R$ (km) & & \multicolumn{3}{c}{$\rho_a$ error} & \multicolumn{2}{c}{$m_a$ error (mV/V)} & CG its & "
                 r"Wall (s)" + (r" & ms/it" if per_iteration else "") + r" \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
                 r" & & Median & 99\% & Max. & Median & 99\% & & " + ("& " if per_iteration else "") + r"\\", r"\midrule"]
        for extent in extents:
            for method in COMPARED:
                m = records[f"{method}-{extent}"]
                e, c = m["errors"]["all"], m["cost"]
                lines.append(f"{extent / 1000:g} & {LABELS[method]} & {sci(e['rho']['median'])} & {sci(e['rho']['p99'])} & "
                             f"{sci(e['rho']['max'])} & {e['ma']['median']:.3f} & {e['ma']['p99']:.3f} & "
                             f"{c['total_iterations'] / c['solves']:.1f} & {c['wall']:.0f}"
                             + (f" & {c['ms_per_iteration']:.2f}" if per_iteration else "") + r" \\")
            if extent != extents[-1]:
                lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(name, lines)

    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lr" + "r" * len(COMPARED) + "}", r"\toprule",
             r"Measure & Threshold & " + " & ".join(LABELS[m] for m in COMPARED) + r" \\", r"\midrule"]
    for measure, thresholds in THRESHOLDS.items():
        title = r"$\rho_a$, 99\%" if measure == "rho_p99" else r"$m_a$, 99\% (mV/V)"
        for index, threshold in enumerate(thresholds):
            cells = []
            for method in COMPARED:
                key = fixed[f"{measure}-{threshold:g}-{method}"]
                cells.append("--" if key is None else
                             f"{records[key]['cost']['wall']:.0f} ({int(key.rsplit('-', 1)[1]) / 1000:g})")
            value = f"{100 * threshold:g}" + r"\%" if measure == "rho_p99" else f"{threshold:g}"
            lines.append(f"{title if index == 0 else ''} & {value} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("example4-fixed.tex", lines)

    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{rlrrrrrr}", r"\toprule",
             r"$p$ & PC & DoFs & CG its & Setup (s) & Solve (s) & Wall (s) & GiB/rank \\", r"\midrule"]
    for order in (2, 3):
        for pc, label in (("lor", "LOR-AMG"), ("amg", "AMG")):
            c = records[f"central-le-p{order}-{pc}-1000"]["cost"]
            lines.append(f"{order} & {label} & {c['dofs']:,} & {c['max_iterations']} & {c['preconditioner_time']:.1f} & "
                         f"{c['solve']:.1f} & {c['wall']:.1f} & {c['memory_gib_per_rank']:.2f} " + r"\\")
        if order == 2:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("example4-solvers.tex", lines)


# Robustness ----------------------------------------------------------------------------------------------------

def robustness_table(out: dict) -> None:
    """Truncation errors with perturbed backgrounds: Example 1 (3-D, 0.5 km), largest errors of the pole-pole and
    dipole-dipole data; Example 3 (1 km), 99th percentile of rho_a; largest m_a errors."""
    lines = [r"\small", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{PP, max.} & \multicolumn{3}{c}{DD, max.} & \multicolumn{3}{c}{$m_a$, max.\ (mV/V)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Background & LS & LE & LI & LS & LE & LI & LS & LE & LI \\", r"\midrule",
             r"\multicolumn{10}{l}{Example 1, 3-D, 0.5 km; DS: PP $9.4\times10^{-2}$, DD $3.6\times10^{-3}$, $m_a$ 9.6} \\"]
    names1 = {"true": "True", "h40": r"$h\times0.8$", "h60": r"$h\times1.2$", "rho1-80": r"$\rho_1\times0.8$",
              "rho1-125": r"$\rho_1\times1.25$", "rho2-500": r"$\rho_2\times0.5$", "rho2-2000": r"$\rho_2\times2$"}
    for background, label in names1.items():
        pp, dd, ip = [], [], []
        for method in ("ls", "le", "li"):
            m = (out["example1-3d"]["runs"][f"{method}-500"] if background == "true"
                 else out["example1-3d"]["robustness"][f"{method}-{background}-500"])
            e = m["errors"]
            pp.append(sci(e["pole"]["rho"]["max"]))
            dd.append(sci(e["dipole"]["rho"]["max"]))
            ip.append(mvv(e["all"]["ma"]["max"]))
        lines.append(f"{label} & " + " & ".join(pp + dd + ip) + r" \\")
    lines.append(r"\midrule")
    ds = out["example3"]["runs"]["ds-dense-1000"]["errors"]["all"]
    lines.append(r" & \multicolumn{3}{c}{$\rho_a$, median} & \multicolumn{3}{c}{$\rho_a$, 99\%} & \multicolumn{3}{c}{$m_a$, max.\ (mV/V)} \\")
    lines.append(r"\multicolumn{10}{l}{Example 3, 1 km; DS: " + f"median {sci(ds['rho']['median'])}, 99" + r"\% " + f"{sci(ds['rho']['p99'])}, "
                 + f"$m_a$ {ds['ma']['max']:.2f}" + r"} \\")
    names3 = {"true": "True", "rho080": r"$\rho\times0.8$", "rho125": r"$\rho\times1.25$", "depth080": r"$z\times0.8$",
              "depth120": r"$z\times1.2$", "edge": "Edge column", "coarse3": "Three layers"}
    for background, label in names3.items():
        median, top, ip = [], [], []
        for method in ("ls", "le", "li"):
            e = out["example3"]["robustness"][f"{method}-{background}-1000"]["errors"]["all"]
            median.append(sci(e["rho"]["median"]))
            top.append(sci(e["rho"]["p99"]))
            ip.append(mvv(e["ma"]["max"]))
        lines.append(f"{label} & " + " & ".join(median + top + ip) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("robustness.tex", lines)


def ip_table(out: dict) -> None:
    """How the errors of the two states combine in m_a (Eq. iperror), Examples 3 and 4 on the 1 km domain, with the
    instantaneous pass using its own Robin coefficients or those of the DC background."""
    rows = []
    for example, base, reference, pattern, methods in (
            ("example1-3d", "results/example1-3d", "results/example1-3d/le-20000", "{m}-500", ("ds", "ls", "le", "li")),
            ("example3", "results/example3", "results/example3/le-dense-200000", "{m}-dense-1000", COMPARED),
            ("example4", "results/example4", "results/example4/le-200000", "{m}-1000", COMPARED)):
        for method in methods:
            rows.append((example, method, "own", f"{base}/{pattern.format(m=method)}", reference))
        for method in ("ls", "le", "li"):
            rows.append((example, method, "dc", f"results/ip-coefficients/{example}/{pattern.format(m=method)}", reference))
    metrics.require([row[3] for row in rows])
    records = {}
    for example, method, coefficients, path, reference in rows:
        records[f"{example}-{method}-{coefficients}"] = metrics.record(path, reference)
    out["ip"] = records
    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{99th percentile} & & \\",
             r"\cmidrule(lr){3-5}",
             r"Example & Treatment & $|\epsilon_{\rm DC}|$ & $|\epsilon_{\rm DC}-\epsilon_{\rm inst}|$ & $|\delta m_a|$ (mV/V) & "
             r"$q$ & Corr. \\", r"\midrule"]
    names = {"example1-3d": "1 (3-D, 0.5 km)", "example3": "3 (2.5-D, 1 km)", "example4": "4 (3-D, 1 km)"}
    for example in names:
        first = True
        for method in COMPARED + ("ls-dc", "le-dc", "li-dc"):
            key = f"{example}-{method}" if method.endswith("-dc") else f"{example}-{method}-own"
            if key not in records:
                continue
            ip = records[key]["ip"]
            label = LABELS[method[:2]] + (r", DC coeff." if method.endswith("-dc") else "")
            lines.append(f"{names[example] if first else ''} & {label} & "
                         f"{sci(ip['dc']['p99'])} & {sci(ip['difference']['p99'])} & "
                         f"{mvv(records[key]['errors']['all']['ma']['p99'])} & {ip['difference_over_dc_median']:.3f} & "
                         f"{ip['correlation']:.4f} " + r"\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("ip-propagation.tex", lines)


def diagnostics(out: dict) -> None:
    """Totals of the boundary diagnostics over all run reports of the paper: quadrature points (times wavenumbers) of
    the shared coefficients and of the per-electrode coefficients, and those clipped or with alpha_0 = 0."""
    totals = {"reports": 0, "boundary_points": 0, "clipped_points": 0, "electrode_points": 0, "electrode_clipped_points": 0,
              "unbounded_points": 0}
    for report in sorted((ROOT / "results").rglob("*.json")):
        if report.name.startswith("provenance") or "validation" in report.parts:
            continue
        data = json.loads(report.read_text())
        totals["reports"] += 1
        for solver in (s for p in data["passes"] for s in p["solvers"]):
            for key in totals:
                if key != "reports":
                    totals[key] += solver.get(key, 0)
    out["diagnostics"] = totals


# Configurations (supplementary material) -----------------------------------------------------------------------

# Example, dimension, results directory, runs over the outer extents, reference run, arrays, largest separation, mesh,
# core cells (m), padding growth; meshes and growth factors as built by scripts/make_examples.py and
# scripts/make_overthrust_octree.py.
CONFIGURATIONS = (
    ("1", "3-D", "results/example1-3d", [f"le-{e}" for e in EXTENTS1 + (20000,)], "le-20000", "PP, DD", 6,
     "tensor", r"$20\times20\times10$", "1.4"),
    ("1", "2.5-D", "results/example1-25d", [f"le-{e}" for e in EXTENTS1 + (20000,)], "le-20000", "PP, DD", 6,
     "tensor", r"$10\times10$", "1.4"),
    ("2", "3-D", "results/example2-3d", [f"le-amg-{e}" for e in (1000, 2000)], "le-amg-20000", "PP, DD", 8,
     "tensor", r"$20\times20\times10$", "1.4"),
    ("2", "2.5-D", "results/example2-25d", [f"le-amg-{e}" for e in (1000, 2000)], "le-amg-20000", "PP, DD", 8,
     "tensor", r"$10\times10$", "1.4"),
    ("3", "2.5-D", "results/example3", [f"le-dense-{e}" for e in EXTENTS], "le-dense-200000", "DD, PD", 10,
     "quadtree", "25--400", "1.3"),
    ("4", "3-D", "results/example4", [f"le-{e}" for e in EXTENTS], "le-200000", "DD, PD", 10,
     "octree", "25--400", "1.3"),
)
RULES = {"log": "Gauss", "layered": "fitted"}
PRECONDITIONERS = {"amg": "AMG", "lor": "LOR-AMG", "direct": "MUMPS"}


def configurations(out: dict) -> None:
    """Surveys, meshes, problem sizes and solver settings of the examples, from the run reports and survey files."""
    def thousands(n: int) -> str:
        return f"{n:,}".replace(",", r"\,")

    def report(path: str) -> dict:
        return json.loads((ROOT / f"{path}.json").read_text())

    columns = {}
    for example, dimension, base, runs, reference, arrays, separation, mesh, cells, growth in CONFIGURATIONS:
        ref = report(f"{base}/{reference}")
        solver = ref["passes"][0]["solvers"][0]
        survey = [ln.split("#")[0].split() for ln in (ROOT / ref["survey"]).read_text().splitlines()]
        survey = [row for row in survey if row]
        rows = sum(1 for ln in (ROOT / f"{base}/{reference}.dat").read_text().splitlines()
                   if ln.strip() and not ln.startswith("#"))
        sizes = [report(f"{base}/{run}")["passes"][0]["solvers"][0]["dofs"] for run in runs]
        everything = [json.loads(p.read_text()) for p in sorted((ROOT / base).glob("*.json"))]
        memory = [r["peak_memory_mib_max"] / 1024 for r in everything]
        columns[f"{example}, {dimension}"] = {
            "electrodes": int(survey[0][0]), "current": solver["n_solves"] // solver["wavenumbers"], "data": rows,
            "refinements": (ref["tx_refinements"], ref["rx_refinements"]),
            "extents_km": [int(run.rsplit("-", 1)[1]) / 1000 for run in runs],
            "reference_km": int(reference.rsplit("-", 1)[1]) / 1000, "dofs": [min(sizes), max(sizes)],
            "reference_dofs": solver["dofs"],
            "wavenumbers": solver["wavenumbers"] if ref["dim"] == 2 else None,
            "rule": RULES.get(ref.get("wavenumber_rule")) if ref["dim"] == 2 else None,
            "orders": sorted({r["order"] for r in everything}),
            "preconditioners": sorted({"MUMPS" if r["solver"] == "mumps" else PRECONDITIONERS[r["preconditioner"]]
                                       for r in everything}),
            "ranks": sorted({r["ranks"] for r in everything}), "memory_gib": [min(memory), max(memory)],
            "arrays": arrays, "largest_separation": separation, "mesh": mesh, "core_cells_m": cells,
            "padding_growth": growth}
    out["configurations"] = columns

    def row(title: str, cells: list[str]) -> str:
        return f"{title} & " + " & ".join(cells) + r" \\"

    c = list(columns.values())
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "l" * len(c) + "}", r"\toprule",
             row("Example", [x[0] for x in CONFIGURATIONS]), row("Dimension", [x[1] for x in CONFIGURATIONS]), r"\midrule",
             row("Electrodes (current)", [f"{x['electrodes']} ({x['current']})" for x in c]),
             row("Data", [str(x["data"]) for x in c]),
             row("Arrays", [x["arrays"] for x in c]),
             row("Largest separation $n$", [str(x["largest_separation"]) for x in c]),
             row("Mesh", [x["mesh"] for x in c]),
             row("Core cells (m)", [x["core_cells_m"] for x in c]),
             row("Padding growth factor", [x["padding_growth"] for x in c]),
             row("Electrode refinements", [f"{x['refinements'][0]}, {x['refinements'][1]}" for x in c]),
             row("Outer extents (km)", [", ".join(f"{e:g}" for e in x["extents_km"]) for x in c]),
             row("Reference extent (km)", [f"{x['reference_km']:g}" for x in c]),
             row("DoFs, outer extents", [f"{thousands(x['dofs'][0])}--" for x in c]),
             row("", [thousands(x["dofs"][1]) for x in c]),
             row("DoFs, reference", [thousands(x["reference_dofs"]) for x in c]),
             row("Wavenumbers", ["--" if x["wavenumbers"] is None else f"{x['wavenumbers']}, {x['rule']}" for x in c]),
             row("Polynomial degree", [", ".join(str(o) for o in x["orders"]) for x in c]),
             row("Preconditioners", [", ".join(x["preconditioners"]) for x in c]),
             row("MPI processes", [", ".join(str(r) for r in x["ranks"]) for x in c]),
             row("GiB per process", [f"{x['memory_gib'][0]:.2f}--{x['memory_gib'][1]:.2f}" for x in c]),
             r"\bottomrule", r"\end{tabular}"]
    write("supp-configurations.tex", lines)


def sanitize(value):
    """Replace non-finite numbers by None so that the metrics file is strict JSON."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    out: dict = {}
    for step in (example1, example2, example3, example4, robustness_table, ip_table, diagnostics, configurations):
        step(out)
    METRICS.write_text(json.dumps(sanitize(out), indent=1, allow_nan=False) + "\n")
    print(f"Wrote {METRICS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
