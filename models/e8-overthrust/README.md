# Examples 3 and 4: DC/IP models derived from the SEG/EAGE Overthrust model

## Source data

- File: SEG/EAGE 3-D Overthrust velocity model ("true model"), `source/overthrust_3D_true_model.h5` (not stored here).
- Download: <https://zenodo.org/records/4252588/files/overthrust_3D_true_model.h5> (Zenodo, doi:10.5281/zenodo.4252588,
  531.3 MB, md5 `ae4082cdcebbc7714eedd382d7925804`).
- Licence: Creative Commons Attribution 4.0 International (<https://creativecommons.org/licenses/by/4.0>),
  "Copyright 1997 Society of Exploration Geophysicists". The Zenodo record asks: "Please DO NOT separate the model and
  data from this license and ALWAYS include the above statement and link with the data."
- Reference: Aminzadeh, F., Brac, J., Kunz, T., 1997. SEG/EAGE 3-D Salt and Overthrust Models. SEG/EAGE 3-D Modeling
  Series, No. 1. Society of Exploration Geophysicists.

## Mapping to resistivity and chargeability

With the velocity v in km/s, rho = 10 (v / 2.18)^6 ohm-m and eta = 0.02 + 0.13 exp(-((v - 3.0) / 0.25)^2)
(functions in `scripts/make_overthrust.py`). The mapping is synthetic and defined for this paper; it is not part of the
original model. The water layer is removed, so that z = 0 is the top of the rock.

## Octree and quadtree models

`octree-section/` (Example 3, 2.5D section at y = 10 km) and `octree-volume/` (Example 4, 3D core) are built in two
steps:

```sh
python3 -I scripts/make_overthrust_octree.py section   # voxels.{txt,bin}, background.txt, root-<extent>.mesh, survey.txt, octree-options.txt, sizes.json
python3 -I scripts/make_overthrust_octree.py volume
code/build/octree_mesh -m models/e8-overthrust/octree-volume/root-500.mesh -g models/e8-overthrust/octree-volume/voxels \
    -bg models/e8-overthrust/octree-volume/background.txt -o models/e8-overthrust/octree-volume/model-500 \
    $(cat models/e8-overthrust/octree-volume/octree-options.txt)
```

`REPRODUCE.md` loops over all extents; on a Slurm cluster `scripts/hpc/slurm/octree_mesh.slurm` does the same.
`voxels.bin` and `model-*` are not stored. The octree construction is deterministic; the md5 sums of the
`model-500.cells` files of the paper runs are `f6b43c8dc94e338ade0ce57c0dbc41a2` (section) and
`edca3ec30b20fca39738659cb88481f2` (volume). `octree-section/backgrounds/` holds the perturbed backgrounds of the
robustness checks (`scripts/make_backgrounds.py`).
