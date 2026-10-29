# Reproducing the examples of the paper

This file describes how to rebuild the models, rerun the four examples and regenerate the tables and figures of the
paper "Layered-earth Robin boundary treatments for multi-electrode finite-element DC resistivity and induced
polarization modelling". The response files and run reports of the published runs are in `results/`, so the
tables and figures can be regenerated without rerunning the forward models (step 4).

## Contents

| Path | Content |
| --- | --- |
| `code/` | C++17 programs `dcipfwd` (forward solver) and `octree_mesh` (octree models), unit tests; see `code/README.md` |
| `scripts/` | Python: inputs of the examples, run definitions, analysis, figures; Slurm scripts in `scripts/hpc/slurm/` |
| `models/` | Inputs: surveys, property and background files; meshes are regenerated (step 2) |
| `results/` | Responses (`.dat`), unit-current potentials (`.pot`) and run reports (`.json`) of all runs of the paper |
| `figures/`, `figures/data/` | Figures (PDF) and the data plotted in them; `paper-metrics.json` holds every number of the tables |

## 1. Software

- `dcipfwd` 1.0.0 (this repository): MPI, MFEM 4.10.0 with hypre, MUMPS and GSLIB, CMake >= 3.20, C++17
  (versions of the paper runs: GCC 14.2.0, Open MPI 4.1.8, hypre 3.0.0, MUMPS 5.8.1).
- Python >= 3.10 with numpy and scipy; matplotlib and SciencePlots for the figures; PyVista for the 3D renderings
  of Example 4; h5py only to read the Overthrust source model. Versions used: Python 3.13, numpy 2.4, scipy 1.17,
  matplotlib 3.10, SciencePlots 2.2, PyVista 0.48, h5py 3.16.

```sh
cd code && cmake -B build -DMFEM_PREFIX=/path/to/mfem && cmake --build build -j 8 && (cd build && ctest)
```

## 2. Inputs

```sh
python3 -I scripts/make_examples.py            # Examples 1 and 2: tensor meshes, properties, surveys
python3 -I scripts/make_backgrounds.py         # background variants of the robustness checks
```

Examples 3 and 4 use the SEG/EAGE Overthrust velocity model (Zenodo, doi:10.5281/zenodo.4252588, CC-BY 4.0,
`overthrust_3D_true_model.h5`, 531 MB, md5 `ae4082cdcebbc7714eedd382d7925804`), saved as
`models/e8-overthrust/source/overthrust_3D_true_model.h5`. The resistivity and chargeability mappings of the paper
are applied by

```sh
python3 -I scripts/make_overthrust_octree.py section    # Example 3: voxels, background, root meshes, survey
python3 -I scripts/make_overthrust_octree.py volume     # Example 4
for case in octree-section octree-volume; do
  d=models/e8-overthrust/$case
  for root in $d/root-*.mesh; do
    extent=${root##*/root-}; extent=${extent%.mesh}
    code/build/octree_mesh -m $root -g $d/voxels -bg $d/background.txt -o $d/model-$extent $(cat $d/octree-options.txt)
  done
done
```

The octree meshes are deterministic: the md5 sums of `model-500.cells` of the paper runs are recorded in
`models/e8-overthrust/README.md`.

## 3. Forward runs

`scripts/paper_runs.py` defines every run of the paper in five groups (`example1`, `example2`, `example3`,
`example4-main`, `example4-aux`) and a sixth group (`ip-coefficients`) that repeats some runs with the Robin
coefficients of the DC background in the instantaneous pass. Each run line gives the output prefix, the input files, the process count and all
options of `dcipfwd`; `python3 -I scripts/paper_runs.py list <group>` prints them.

```sh
python3 -I scripts/paper_runs.py groups                     # run counts
python3 -I scripts/paper_runs.py run example1 --ranks 8     # local run; finished runs are skipped
```

The published runs used exclusive cluster nodes with two 128-core AMD EPYC 9755 processors and 768 GB, one node per
group and runs in sequence, with 64 MPI processes in Examples 1 to 3 and 256 in Example 4. On a Slurm cluster,
`scripts/hpc/slurm/submit_all.sh` submits the build, the unit tests and the groups as jobs pinned to one build (site
modules and the partition go in `scripts/hpc/slurm/site.sh`); the run lists of the paper are in
`scripts/hpc/slurm/lists/`. Wall times of the groups on such a node: example1, 13 min; example2, 28 min; example3,
36 min; example4-main, 1 h 40 min; example4-aux, 50 min; ip-coefficients, 15 min. Examples 1 and 3 also run on a laptop with 8 processes; Example 4 needs
up to about 300 GB of memory in total (the 400 km reference).

Labels of the boundary treatments in the output names: `ds` homogeneous coefficient with a shared reference, `de`
homogeneous coefficient with per-electrode references, `ls` layered coefficient with a shared reference, `le` layered
coefficient with per-electrode references, `li` shared layered coefficient with the layered potential of each
electrode as Robin data, `lm` one matrix per electrode (check of `le`).

## 4. Tables and figures

```sh
python3 -I scripts/figures/paper_tables.py     # figures/data/paper-metrics.json and manuscript/tables/*.tex (supp-*: supplement)
python3 -I scripts/octree_metrics.py           # model approximation errors of the octree meshes (needs model-500.cells)
for f in fig_beta fig_schematic fig_example1 fig_example3 fig_example4 fig_cost; do python3 -I scripts/figures/$f.py; done
# a subset rerun into another directory, for comparison with results/:
python3 -I scripts/paper_runs.py run example1 --match -500 --ranks 8 --into check
```

| Figure or table | Script | Results |
| --- | --- | --- |
| Fig. 1 (beta) | `fig_beta.py` | analytic |
| Figs. 2, 3 (Example 1) | `fig_example1.py` | `results/example1-*` |
| Table 1 (Example 2) | `paper_tables.py` | `results/example2-*` |
| Figs. 4, 5 (Example 3) | `fig_example3.py` | `results/example3` |
| Figs. 6, 7 (Example 4) | `fig_example4.py` | `results/example4` |
| Fig. 8 and Table 2 (accuracy and cost) | `fig_cost.py`, `paper_tables.py` | `results/example1-3d`, `results/example3`, `results/example4` |
| Supplementary Tables S1, S2 (configurations; Example 1) | `paper_tables.py` | `results/example*` |
| Supplementary Figs. S1, S3 and Table S6 (Example 3: pseudosections; wavenumber rules) | `fig_example3.py`, `paper_tables.py` | `results/example3` |
| Supplementary Fig. S2 and Tables S3, S7 (Example 4: responses; errors; solvers) | `fig_example4.py`, `paper_tables.py` | `results/example4` |
| Supplementary Table S4 (apparent chargeability) | `paper_tables.py` | `results/example1-3d`, `results/example3`, `results/example4`, `results/ip-coefficients` |
| Supplementary Table S5 (background perturbations) | `paper_tables.py` | `results/robustness` |
| Supplementary Fig. S4 (schematic) | `fig_schematic.py` | none |

The runs of the five example groups used one build of the code; the `ip-coefficients` group used a later build that adds
only the option `-ip-coefficients`; version 1.0.0 adds to it only timing output (`rule_time`) and tests. The released
code (version 1.0.0, this repository) was checked against these results: built on the same cluster (all unit tests
passed), it reran the 141 runs of the `example1` group and DS, LE and LI on the 1 and 10 km domains of Examples 3 and 4
(run list `scripts/hpc/slurm/lists/verification.txt`). The voltages and apparent chargeabilities of all 153 runs are
bitwise identical to those in `results/` (`results/validation/2026-10-28-release-verification.json`). A rerun of the
Example 1 runs at 0.5 and 5 km on a laptop (8 processes, other compiler and library versions) agrees with `results/` to
1.1e-7 in relative terms (`results/validation/2026-10-28-local-reproduction.json`).

Truncation errors are differences from a reference on the same core mesh with a far boundary (20 km in Examples 1
and 2, 200 km in Examples 3 and 4); total errors (Example 1) are differences from the analytic two-layer responses.
Times are the forward-pass wall times of the DC and instantaneous solutions reported in the `.json` files, measured on
the first process; they exclude program start, file input and output, and the 2.5D wavenumber-rule set-up, which
version 1.0.0 reports as `rule_time` (`results/validation/2026-10-28-rule-time/`).
