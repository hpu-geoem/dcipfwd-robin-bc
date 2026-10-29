"""Forward runs of the four examples of the paper, as one list per job group.

    python3 -I scripts/paper_runs.py groups                       # list the job groups and their run counts
    python3 -I scripts/paper_runs.py list GROUP [--ranks N]        # print the run lines of a group
    python3 -I scripts/paper_runs.py run GROUP [--ranks N] [--dry]  # run a group locally, skipping finished runs
    python3 -I scripts/paper_runs.py run example1 --match le-500 --into check   # a subset, written under check/

Groups example1 to example4-aux are the runs of the examples; ip-coefficients repeats some of them with the Robin
coefficients of the DC background in the instantaneous pass. A run line is "output mesh properties survey ranks options...", with paths relative to the project root; the cluster
job scripts/hpc/slurm/run_list.slurm executes the same lines. Locally, DCIPFWD_BINARY selects the program (default
code/build/dcipfwd) and MPIEXEC the launcher (default: mpiexec on the PATH).
--ranks overrides the process count of every run (the paper used 64 for examples 1-3 and 256 for example 4).

Method labels (Section 4 of the paper):
  ds  homogeneous (Dey-Morrison) coefficient, shared reference      de  homogeneous coefficient, per-electrode references
  ls  layered coefficient, shared reference                         le  layered coefficient, per-electrode references
  li  shared layered coefficient with the layered potential of each electrode as inhomogeneous Robin data
  lm  layered coefficient, one matrix and preconditioner per electrode (check of le)
The shared reference is the centroid of the current electrodes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METHODS = {"ds": ["-bc", "dey-morrison"], "de": ["-bc", "dey-morrison", "-ref", "electrode"], "ls": [],
           "le": ["-ref", "electrode"], "li": ["-ref", "electrode-data"], "lm": ["-ref", "electrode-matrix"]}
COMPARED = ("ds", "de", "ls", "le", "li")

# Background variants of the robustness checks: the true model is unchanged, only the background that defines the
# boundary condition is perturbed (files written by scripts/make_backgrounds.py).
EXAMPLE1_BACKGROUNDS = ("h40", "h60", "rho1-80", "rho1-125", "rho2-500", "rho2-2000")
EXAMPLE3_BACKGROUNDS = ("rho080", "rho125", "depth080", "depth120", "edge", "coarse3")


def run(output: str, model: str, mesh: str, prop: str, survey: str, ranks: int, options: list[str]) -> tuple:
    return (output, f"models/{model}/{mesh}", f"models/{model}/{prop}", f"models/{model}/{survey}", ranks, options)


def example1() -> list[tuple]:
    """Two-layer earth, 3D and 2.5D: all methods on all extents, background perturbations and the 2.5D decay switch."""
    common = ["-order", "2", "-rtx", "3", "-rrx", "2", "-solver", "cg", "-pc", "amg", "-rtol", "1e-10", "-maxit", "1000",
              "-pot"]
    runs = []
    for case, model in (("example1-3d", "ex01-basement-3d"), ("example1-25d", "ex02-basement-25d")):
        for extent in (500, 1000, 2000, 5000, 20000):
            for method in COMPARED + ("lm",):
                runs.append(run(f"results/{case}/{method}-{extent}", model, f"model-{extent}.mesh", "model.prop",
                                "survey.txt", 64, common + METHODS[method]))
        for extent in (500, 1000):
            for background in EXAMPLE1_BACKGROUNDS:
                for method in ("ls", "le", "li"):
                    runs.append(run(f"results/robustness/{case}/{method}-{background}-{extent}", model,
                                    f"model-{extent}.mesh", "model.prop", "survey.txt", 64,
                                    common + METHODS[method] + ["-bg", f"models/{model}/backgrounds/{background}.txt"]))
    for switch in (10, 20, 25):
        for method in ("ls", "le", "li"):
            runs.append(run(f"results/robustness/example1-25d/{method}-kl{switch}-500", "ex02-basement-25d",
                            "model-500.mesh", "model.prop", "survey.txt", 64,
                            common + METHODS[method] + ["-kl-switch", str(switch)]))
    return runs


def example2() -> list[tuple]:
    """48-electrode line, 3D and 2.5D: accuracy and cost of the solution strategies."""
    common = ["-order", "2", "-rtx", "3", "-rrx", "2", "-rtol", "1e-10", "-maxit", "1000", "-pot"]
    variants = {"ds-amg": METHODS["ds"] + ["-pc", "amg"], "de-amg": METHODS["de"] + ["-pc", "amg"],
                "ls-amg": ["-pc", "amg"], "ls-mumps": ["-solver", "mumps"],
                "le-amg": METHODS["le"] + ["-pc", "amg"], "le-mumps": METHODS["le"] + ["-pc", "direct"],
                "li-amg": METHODS["li"] + ["-pc", "amg"], "li-mumps": METHODS["li"] + ["-solver", "mumps"],
                "lm-amg": METHODS["lm"] + ["-pc", "amg"], "lm-mumps": METHODS["lm"] + ["-solver", "mumps"]}
    runs = []
    for case, model in (("example2-3d", "ex03-line48-3d"), ("example2-25d", "ex04-line48-25d")):
        runs.append(run(f"results/{case}/le-amg-20000", model, "model-20000.mesh", "model.prop", "survey.txt", 64,
                        common + variants["le-amg"]))
        for extent in (1000, 2000):
            for name, options in variants.items():
                runs.append(run(f"results/{case}/{name}-{extent}", model, f"model-{extent}.mesh", "model.prop",
                                "survey.txt", 64, common + options))
    return runs


def octree(case: str, extent: int, order: int = 2) -> list[str]:
    return ["-order", str(order), "-rtx", "2", "-rrx", "1", "-rtol", "1e-10", "-maxit", "4000", "-bg",
            f"models/e8-overthrust/{case}/background.txt"]


def example3() -> list[tuple]:
    """Overthrust section in 2.5D (quadtree): truncation, wavenumber rules, configurations and robustness."""
    model, out = "e8-overthrust/octree-section", "results/example3"
    dense = ["-kq", "layered", "-kppd", "6"]

    def section(output: str, extent: int, options: list[str], order: int = 2) -> tuple:
        return run(output, model, f"model-{extent}.mesh", f"model-{extent}.prop", "survey.txt", 64,
                   octree("octree-section", extent, order) + ["-pot"] + options)

    runs = [section(f"{out}/le-dense-200000", 200000, METHODS["le"] + dense),
            section(f"{out}/le-dense-400000", 400000, METHODS["le"] + dense),
            section(f"{out}/le-dense-p3-200000", 200000, METHODS["le"] + dense, order=3),
            section(f"{out}/le-layered10-200000", 200000, METHODS["le"] + ["-kq", "layered", "-kppd", "10"]),
            section(f"{out}/le-log10-200000", 200000, METHODS["le"] + ["-kq", "log", "-kppd", "10"])]
    for extent in (500, 1000, 2000, 5000, 10000):
        for method in COMPARED:
            runs.append(section(f"{out}/{method}-dense-{extent}", extent, METHODS[method] + dense))
    for rule in ("log", "layered", "layered-joint", "layered-separate"):
        for ppd in (2, 3, 4):
            runs.append(section(f"{out}/le-{rule}{ppd}-200000", 200000, METHODS["le"] + ["-kq", rule, "-kppd", str(ppd)]))
    # Practical configurations with 25 wavenumbers.
    for extent in (500, 1000, 2000, 5000, 10000):
        for method in COMPARED:
            for rule in ("log", "layered"):
                runs.append(section(f"{out}/{method}-{rule}4-{extent}", extent,
                                    METHODS[method] + ["-kq", rule, "-kppd", "4"]))
    # The logarithmic rule keeps the weights independent of the perturbed background.
    fixed = ["-kq", "log", "-kppd", "6"]
    for background in ("true",) + EXAMPLE3_BACKGROUNDS:
        file = f"models/{model}/background.txt" if background == "true" else f"models/{model}/backgrounds/{background}.txt"
        for method in ("ls", "le", "li"):
            runs.append(run(f"results/robustness/example3/{method}-{background}-1000", model, "model-1000.mesh",
                            "model-1000.prop", "survey.txt", 64,
                            octree("octree-section", 1000)[:-2] + ["-pot", "-bg", file] + METHODS[method] + fixed))
    for switch in (10, 20, 25):
        for method in ("ls", "le", "li"):
            runs.append(section(f"results/robustness/example3/{method}-kl{switch}-1000", 1000,
                                METHODS[method] + dense + ["-kl-switch", str(switch)]))
    return runs


def example4(part: str) -> list[tuple]:
    """Overthrust volume in 3D (octree): compact domains of all methods on one node, references and solvers on another."""
    model, out = "e8-overthrust/octree-volume", "results/example4"

    def volume(output: str, extent: int, options: list[str], survey: str = "survey.txt", order: int = 2) -> tuple:
        return run(output, model, f"model-{extent}.mesh", f"model-{extent}.prop", survey, 256,
                   octree("octree-volume", extent, order) + options)

    if part == "main":
        runs = []
        for extent in (500, 1000, 2000, 5000, 10000):
            for method in COMPARED:
                runs.append(volume(f"{out}/{method}-{extent}", extent,
                                   METHODS[method] + ["-pc", "lor"] + (["-pot"] if extent == 1000 else [])))
        return runs
    runs = [volume(f"{out}/le-200000", 200000, METHODS["le"] + ["-pc", "lor"]),
            volume(f"{out}/le-400000", 400000, METHODS["le"] + ["-pc", "lor"])]
    for order in (2, 3):
        for pc in ("lor", "amg"):
            runs.append(volume(f"{out}/central-le-p{order}-{pc}-1000", 1000, METHODS["le"] + ["-pc", pc],
                               survey="survey-central.txt", order=order))
    # Repeats on a second node bound the node-to-node variation of the times.
    for method, extent in (("ls", 1000), ("le", 1000), ("li", 1000), ("ds", 5000)):
        runs.append(volume(f"{out}/repeat-{method}-{extent}", extent, METHODS[method] + ["-pc", "lor"]))
    return runs


def ip_coefficients() -> list[tuple]:
    """Instantaneous pass with the Robin coefficients of the DC background (-ip-coefficients dc), examples 1, 3 and 4."""
    dc = ["-ip-coefficients", "dc"]
    runs = []
    common1 = ["-order", "2", "-rtx", "3", "-rrx", "2", "-solver", "cg", "-pc", "amg", "-rtol", "1e-10", "-maxit", "1000"]
    for case, model in (("example1-3d", "ex01-basement-3d"), ("example1-25d", "ex02-basement-25d")):
        for method in ("ls", "le", "li"):
            runs.append(run(f"results/ip-coefficients/{case}/{method}-500", model, "model-500.mesh", "model.prop",
                            "survey.txt", 64, common1 + METHODS[method] + dc))
    model = "e8-overthrust/octree-section"
    for extent in (500, 1000, 2000):
        for method in ("ls", "le", "li"):
            runs.append(run(f"results/ip-coefficients/example3/{method}-dense-{extent}", model, f"model-{extent}.mesh",
                            f"model-{extent}.prop", "survey.txt", 64,
                            octree("octree-section", extent) + METHODS[method] + ["-kq", "layered", "-kppd", "6"] + dc))
    model = "e8-overthrust/octree-volume"
    for method in ("ls", "le", "li"):
        runs.append(run(f"results/ip-coefficients/example4/{method}-1000", model, "model-1000.mesh", "model-1000.prop",
                        "survey.txt", 256, octree("octree-volume", 1000) + METHODS[method] + ["-pc", "lor"] + dc))
    return runs


GROUPS = {"example1": example1, "example2": example2, "example3": example3,
          "example4-main": lambda: example4("main"), "example4-aux": lambda: example4("aux"),
          "ip-coefficients": ip_coefficients}


def line(entry: tuple, ranks: int | None, into: str | None = None) -> str:
    output, mesh, prop, survey, default_ranks, options = entry
    if into:
        output = str(Path(into) / Path(output).relative_to("results"))
    return " ".join([output, mesh, prop, survey, str(ranks or default_ranks)] + options)


def binary() -> Path:
    path = Path(os.environ.get("DCIPFWD_BINARY", ROOT / "code" / "build" / "dcipfwd")).expanduser().resolve()
    if not path.exists():
        sys.exit(f"Build the code first: {path} is missing (see code/README.md).")
    return path


def launcher() -> list[str]:
    if "MPIEXEC" in os.environ:
        return shlex.split(os.environ["MPIEXEC"])
    return [shutil.which("mpiexec") or "mpiexec"]


def run_locally(entries: list[tuple], ranks: int | None, dry: bool, into: str | None) -> None:
    program = binary()
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    for entry in entries:
        output, mesh, prop, survey, default_ranks, options = entry
        if into:
            output = str(Path(into) / Path(output).relative_to("results"))
        out = ROOT / output
        if (out.parent / f"{out.name}.json").exists():
            continue
        command = launcher() + ["-n", str(ranks or default_ranks), str(program), "-m", mesh, "-p", prop, "-s", survey,
                                "-o", output] + options
        print(" ".join(command), flush=True)
        if dry:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.parent / f"{out.name}.log", "w") as log:
            log.write(f"# binary sha256 {digest}\n# command {' '.join(command)}\n")
            log.flush()
            subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("groups", "list", "run"))
    parser.add_argument("group", nargs="?", choices=sorted(GROUPS))
    parser.add_argument("--ranks", type=int, help="process count of every run (default: as in the paper)")
    parser.add_argument("--dry", action="store_true", help="print the commands only")
    parser.add_argument("--match", help="only runs whose output path contains this text")
    parser.add_argument("--into", help="write the outputs under this directory instead of results/")
    args = parser.parse_args()
    if args.action == "groups":
        for name, make in GROUPS.items():
            print(f"{name}: {len(make())} runs")
        return
    if not args.group:
        parser.error("a group is required")
    entries = [entry for entry in GROUPS[args.group]() if not args.match or args.match in entry[0]]
    if args.action == "list":
        print("\n".join(line(entry, args.ranks, args.into) for entry in entries))
    else:
        run_locally(entries, args.ranks, args.dry, args.into)


if __name__ == "__main__":
    main()
