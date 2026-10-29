# dcipfwd

Finite-element forward modelling of DC resistivity and integral-chargeability induced-polarization (IP) data in 3D
and 2.5D, with Robin conditions on the truncation boundary derived from the Green's function of a horizontally
layered background.

- Continuous H1 elements of any order on MFEM meshes (hexahedra, tetrahedra, quadrilaterals, triangles), including
  nonconforming octree (quadtree) meshes; z points down; the mesh covers the ground only (no air).
- Total-potential formulation for point electrodes with local refinement around the electrodes.
- Robin coefficient `-dG/dn / G` of the layered background (`-bc layered`) or of a homogeneous half-space
  (`-bc dey-morrison`), in 3D and for every wavenumber in 2.5D.
- Treatment of the current electrodes (`-ref`): one shared reference source and one matrix for all electrodes;
  per-electrode references, whose matrices differ from the shared matrix by a boundary mass term and share its
  preconditioner; or the shared matrix with the layered potential of each electrode as inhomogeneous Robin data.
- 2.5D: cosine transform along the strike with a logarithmic Gauss rule in the wavenumber, or the same nodes with
  weights fitted to the layered background of the DC state, of both states jointly, or of each state separately.
- Solvers: conjugate gradients (CG) preconditioned by BoomerAMG on the high-order matrix, BoomerAMG on the
  low-order-refined (LOR) matrix, or the MUMPS factorization of the shared matrix; or MUMPS as a direct solver.
- IP: second solve with the instantaneous conductivity sigma / (1 - eta); m_a = 1 - dV_inst / dV.
- `octree_mesh`: builds octree models from voxel grids of log10 resistivity and chargeability.

Version 1.0.0. Licence: MIT (`LICENSE`); the bundled archives in `third_party/` keep their own licences (fmt: MIT;
Catch2: BSL-1.0). Repository: <https://github.com/hpu-geoem/dcipfwd-robin-bc>.

## Build

Requirements: CMake >= 3.20, a C++17 compiler, MPI, and MFEM (tested with 4.10.0) built with MPI, hypre, METIS, MUMPS
and GSLIB (LAPACK for one unit test). fmt 12.1.0 and Catch2 3.14.0 (tests only) are bundled in `third_party/`.

### Installing MFEM

The simplest way is [Spack](https://spack.io), which builds MFEM and all its dependencies (hypre, METIS, MUMPS with
ScaLAPACK, GSLIB, BLAS/LAPACK):

```sh
spack install mfem@4.10.0 +mpi +mumps +gslib +lapack
export MFEM_DIR=$(spack location -i mfem)
```

Alternatively, build MFEM with CMake against an installed MUMPS (version 5.1 or later, parallel version, from a package
manager or <https://mumps-solver.org>) and METIS 5; MFEM fetches hypre and GSLIB itself:

```sh
git clone --depth 1 --branch v4.10 https://github.com/mfem/mfem.git
cmake -S mfem -B mfem-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/mfem \
    -DMFEM_USE_MPI=YES -DMFEM_FETCH_HYPRE=YES -DMETIS_DIR=/path/to/metis \
    -DMFEM_USE_GSLIB=YES -DMFEM_FETCH_GSLIB=YES -DMFEM_USE_LAPACK=YES \
    -DMFEM_USE_MUMPS=YES -DMUMPS_DIR=/path/to/mumps -DScaLAPACK_DIR=/path/to/scalapack/lib/cmake/scalapack-<version> \
    -DMUMPS_REQUIRED_PACKAGES="MPI;MPI_Fortran;METIS;ScaLAPACK;LAPACK;BLAS"
cmake --build mfem-build -j 8 && cmake --install mfem-build
export MFEM_DIR=$HOME/opt/mfem
```

`MUMPS_REQUIRED_PACKAGES` lists the libraries MUMPS was built with (add `ParMETIS` or `OpenMP` if needed). MFEM's
`INSTALL` file describes all options, including the GNU make build.

### Building dcipfwd

Tested configurations:

| Machine | Compiler, MPI | MFEM, hypre, MUMPS |
| --- | --- | --- |
| Apple M5 Pro, macOS 26.6 (development, unit tests) | Apple clang 21.0, Open MPI 5.0.10 | 4.10.0, 3.1.0, 5.9.1 |
| 2 x AMD EPYC 9755, Linux cluster node (all paper runs) | GCC 14.2.0, Open MPI 4.1.8 | 4.10.0, 3.0.0, 5.8.1 |

```sh
cmake -B build -DMFEM_PREFIX=$MFEM_DIR   # or set MFEM_DIR in the environment, or use a CMakeUserPresets.json preset
cmake --build build -j 8
cd build && ctest
```

MFEM may provide either its CMake package (`MFEMConfig.cmake`) or only its makefile configuration
(`share/mfem/config.mk`, read by `cmake/MFEMConfigMk.cmake`). `-DDCIPFWD_BUILD_TESTS=OFF` skips the tests.

## Run

```sh
mpiexec -n 8 build/dcipfwd -m model.mesh -p model.prop -s survey.txt -o result \
    -order 2 -rtx 3 -rrx 2 -bc layered -ref electrode -pc lor -rtol 1e-10
```

`build/dcipfwd -h` lists all options. The main ones:

| Option | Meaning |
| --- | --- |
| `-m`, `-p`, `-s`, `-o` | Mesh, cell properties, survey, output prefix |
| `-bg corner` / `-bg file` | Layered background: the bounding-box side (2D) or corner (3D) column that differs least from the model, or rows `top rho [eta]` |
| `-bc layered` / `-bc dey-morrison` | Robin coefficient of the layered background or of a homogeneous half-space |
| `-ref centroid` / `-ref x,y,z` | One shared reference source (default: centroid of the current electrodes) |
| `-ref electrode` | Per-electrode references; the shared preconditioner serves all electrodes |
| `-ref electrode-data` | Shared coefficient; the layered potential of each electrode enters as Robin data |
| `-ref electrode-matrix` | Per-electrode references with one matrix and preconditioner per electrode (check) |
| `-robin-table auto\|always\|never` | Per-electrode layered values from interpolation tables (auto: when cheaper) |
| `-order p` | Polynomial degree |
| `-rtx n`, `-rrx n`, `-ru n` | Refinements around current and potential electrodes; uniform refinements |
| `-solver cg\|mumps`, `-pc amg\|lor\|direct` | Solver and CG preconditioner |
| `-rtol`, `-maxit` | Relative residual tolerance and iteration limit of CG |
| `-amg-theta`, `-blr-tol` | BoomerAMG strength threshold (default 0.6); MUMPS block low-rank tolerance (default 0: exact) |
| `-kmin`, `-kmax`, `-kppd` | 2.5D wavenumber range (default 1e-3 / r_max, 15 / r_min) and Gauss points per decade |
| `-kq log\|layered\|layered-joint\|layered-separate` | 2.5D weights: Gauss; fitted to the DC background; to both states jointly; to each state |
| `-kl-switch x` | 2.5D: k L above which Robin coefficients use the source and its image only (default 15) |
| `-ip auto\|yes\|no` | Instantaneous-conductivity pass (auto: when a cell has eta > 0) |
| `-surface z` | Image plane of the geometric factor (default: top of the background) |
| `-pot` | Also write the unit-current potentials at the potential electrodes |
| `-v` | Log every linear solve |

Solver settings fixed in the code: CG starts from zero; BoomerAMG is used as one V-cycle with the MFEM defaults
(HMIS coarsening, one level of aggressive coarsening, extended+i interpolation with at most four entries per row,
one sweep of l1-scaled symmetric hybrid Gauss-Seidel), which is symmetric and positive definite as required by CG
(checked in `test_spectral_bound`); MUMPS factorizes the symmetric positive-definite matrix.

### Boundary treatments

The paper compares five treatments; each combines a coefficient with a treatment of the current electrodes:

| Label | Options | Coefficient | Electrodes |
| --- | --- | --- | --- |
| DS | `-bc dey-morrison` | homogeneous half-space | shared reference |
| DE | `-bc dey-morrison -ref electrode` | homogeneous half-space | per-electrode references, shared preconditioner |
| LS | `-bc layered` | layered background | shared reference |
| LE | `-bc layered -ref electrode` | layered background | per-electrode references, shared preconditioner |
| LI | `-bc layered -ref electrode-data` | layered background (shared) | data `dG_e/dn + alpha_0 G_e` of each electrode |

With LE, the matrix of electrode e is A_0 + B_(alpha_e - alpha_0), applied without assembly, and CG uses the
preconditioner of A_0. With LI, every electrode solves with A_0 (one factorization serves all with MUMPS); the right-hand
side adds the boundary integral of sigma (dG_e/dn + alpha_0 G_e), which makes the condition exact for the background
potential G_e of the electrode.

## Layered background

The layered coefficients and boundary data use a horizontally layered background (`-bg`). `-bg corner` (the default)
takes the column at the side (2D) or vertical edge (3D) of the mesh bounding box that differs least from the model. This
is adequate when the model is laterally uniform near the boundary, for example when the padding carries a prescribed
layered background, but it is not guaranteed to represent the region. For strongly heterogeneous models, a laterally
averaged or regionally constrained layered background is preferable to an unrepresentative single edge column: use depth
intervals that resolve the main layering, the lateral mean of log10 resistivity and of chargeability of the model in
each interval, and pass the file with `-bg background.txt`. The Overthrust examples of the paper use 400 m intervals.
For an octree model, the averages follow from `model.cells` of `octree_mesh`:

```python
import numpy as np
cells = np.loadtxt("model.cells")  # per leaf: lower corner, upper corner, log10 resistivity, chargeability, core flag
dim = (cells.shape[1] - 3) // 2
lower, upper, core = cells[:, :dim], cells[:, dim:2 * dim], cells[:, -1] == 1
volume = np.prod(upper - lower, axis=1)
depth = 0.5 * (lower[:, -1] + upper[:, -1])  # z of the cell centre, positive down
tops = np.arange(0.0, 4800.0, 400.0)         # layer tops (m)
with open("background.txt", "w") as file:
    file.write("# top resistivity chargeability\n")
    for top, bottom in zip(tops, np.append(tops[1:], np.inf)):
        inside = core & (depth >= top) & (depth < bottom)
        log_rho = np.average(cells[inside, -3], weights=volume[inside])
        eta = np.average(cells[inside, -2], weights=volume[inside])
        file.write(f"{top:g} {10.0 ** log_rho:.6g} {eta:.6g}\n")
```

Applied to the octree models of Examples 3 and 4, this reproduces their background files. The sensitivity of the
results to the background is discussed in the paper (Section 4.7).

## Files

- Mesh (`-m`): any MFEM mesh. 3D uses (x, y, z); 2D is the (y, z) section of a 2.5D model with strike along x.
  Boundary faces whose outward normal points up are the insulating ground surface; all others are the truncation
  boundary.
- Properties (`-p`): rows `attribute resistivity [chargeability]`; `#` starts a comment.
- Survey (`-s`): the electrode count, `x y z` rows, the measurement count, `a b m n` rows (0-based electrode
  indices, `-1` for a pole at infinity). In 2.5D all electrodes have the same x; fitted weights also need one depth.
- Background (`-bg`): rows `top resistivity [chargeability]` with tops increasing downwards.
- `<prefix>.dat`: `a b m n dV K rho_a [dV_inst m_a]` per measurement (dV in V per A).
- `<prefix>.pot` (`-pot`): `current receiver V [V_inst]` for every current and potential electrode pair.
- `<prefix>.json`: version, options, background, wavenumber rules with their a-priori errors against the layered
  background of each state and the time of their set-up (`rule_time`, 2.5D), peak memory per rank and in total, and per pass and solver: cells, DoFs,
  truncation faces, wavenumbers, times (refine, fe, boundary, assemble, preconditioner, solve, electrode_boundary;
  maximum over ranks), layered-potential evaluations and tables, CG iterations, and boundary diagnostics:
  `boundary_points` and `clipped_points` (quadrature points of the truncation faces times wavenumbers, and those where
  the shared ratio -dG/dn / G was negative and set to zero), the same for the per-electrode coefficients
  (`electrode_points`, `electrode_clipped_points`), `unbounded_points` (alpha_e > 0 where alpha_0 = 0), and the range
  `ratio_min`, `ratio_max` of alpha_e / alpha_0 that bounds the condition number of the per-electrode systems.

## Octree models

```sh
build/octree_mesh -m root.mesh -g voxels -bg background.txt -o model \
    -tol-rho 0.1 -tol-eta 0.02 -min-size 1000:25,2500:50,1e30:100 -surface-box 8000,12000,8000,12000 -surface-size 50
```

Root cells with attribute 1 (the core) must align with the voxel grid (`voxels.txt`: dimension, origin, spacing, shape;
`voxels.bin`: log10 resistivity, then chargeability, float64, axis 0 fastest). A core cell is split while the standard
deviation of log10 resistivity or chargeability over its voxels exceeds `-tol-rho`/`-tol-eta` and it is larger than the
minimum size for the depth of its top (`depth:size` rules, first match), or while it is a surface cell in
`-surface-box` larger than `-surface-size`. Neighbours differ by at most `-nc-limit` levels. Core leaves take the mean
log10 resistivity and chargeability of their voxels, other cells the background layer at their centre. Outputs:
`model.mesh`, `model.prop` and `model.cells` (bounding box, log10 resistivity, chargeability and core flag per leaf).

## Examples of the paper

`../REPRODUCE.md` describes how to build the models, run the four examples (`../scripts/paper_runs.py`, locally or
with the Slurm scripts in `../scripts/hpc/slurm`) and regenerate the tables and figures.

## Tests

`ctest` runs every test on one or two MPI processes (about one minute in total).

| Test | Content |
| --- | --- |
| `test_integral_transforms` | Bessel functions, wavenumber quadrature, layered potential against image series (3D and 2.5D) |
| `test_dc_3d` | Half-space (surface, buried), two-layer earth, layered against homogeneous truncation, reciprocity, solvers |
| `test_dc_25d` | Half-space, two-layer truncation and wavenumber resolution, 2.5D against a 3D extrusion, solvers |
| `test_ip` | Uniform chargeability; layered apparent chargeability against the analytic two-layer response; instantaneous coefficients and boundary data of the instantaneous background |
| `test_model_survey` | Background columns (also under topography and sloping interfaces), survey and background files |
| `test_dc_reference` | Per-electrode references against one matrix per electrode; DS, DE, LS, LE and LI against a far boundary; Robin tables against direct evaluation, ln G and the decay switch; clipped coefficients and alpha_0 = 0 |
| `test_wavenumber_rule` | Layered 2.5D reference; weights fitted to the DC state, to both states and to each state |
| `test_octree_model` | Octree refinement, background cells, nonconforming mesh output, quadtree two-layer 2.5D potentials |
| `test_spectral_bound` | Generalized eigenvalues of per-electrode and shared operators within the bound of the coefficient ratio; symmetry and positive definiteness of the AMG and LOR-AMG preconditioners on a 3D mesh with hanging nodes |

## Authors

Ce Qin and Songyan Wei (Henan Polytechnic University) and Jun Zhou (University of Electronic Science and Technology of
China). Contact: Ce Qin (ce.qin@hpu.edu.cn), Jun Zhou (zj397@uestc.edu.cn).
