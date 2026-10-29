# dcipfwd: layered-earth Robin boundary treatments for DC resistivity and IP modelling

Code, inputs, results and figures of the paper

> Ce Qin, Songyan Wei, Jun Zhou. Layered-earth Robin boundary treatments for multi-electrode finite-element DC
> resistivity and induced polarization modelling. Submitted to Computers & Geosciences.

- `code/`: the C++17 forward modelling program `dcipfwd` (version 1.0.0) and the octree tool `octree_mesh`, with unit
  tests; build, options and file formats in [`code/README.md`](code/README.md).
- [`REPRODUCE.md`](REPRODUCE.md): how to rebuild the models, rerun every example and regenerate every table and figure.
- `scripts/`, `models/`, `results/`, `figures/`: inputs, run definitions, outputs and plots of the four examples.

```sh
git clone https://github.com/hpu-geoem/dcipfwd-robin-bc.git
cd dcipfwd-robin-bc/code
cmake -B build -DMFEM_PREFIX=/path/to/mfem && cmake --build build -j 8 && (cd build && ctest)
```

The results in `results/` were computed with this version (see `REPRODUCE.md` for the builds and the verification
runs). The code is distributed under the MIT licence (`LICENSE`). Examples 3 and 4 derive from the SEG/EAGE
Overthrust model (Aminzadeh, Brac and Kunz, 1997; Zenodo, doi:10.5281/zenodo.4252588), Copyright 1997 Society of
Exploration Geophysicists, distributed under CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0).

Contact: Ce Qin (ce.qin@hpu.edu.cn), Jun Zhou (zj397@uestc.edu.cn).
