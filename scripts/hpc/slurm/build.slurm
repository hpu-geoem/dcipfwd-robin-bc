#!/bin/bash
#SBATCH -J dcip-build
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Build into code/build-next, run the unit tests, and publish the programs as code/bin/<program>-<sha> with the links
# code/bin/dcipfwd and code/bin/octree_mesh only if everything passed. code/bin/release-<job> lists the published paths;
# jobs given RELEASE=<that file> use exactly these programs. Submit from scripts/hpc/slurm: sbatch build.slurm
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"
set -u
cd $ROOT/code
rm -rf build-next
cmake -B build-next -DCMAKE_BUILD_TYPE=Release -DMFEM_PREFIX=$MFEM_DIR
cmake --build build-next -j 32
(cd build-next && ctest --output-on-failure)
mkdir -p bin
manifest=bin/release-${SLURM_JOB_ID}
: > $manifest.tmp
for program in dcipfwd octree_mesh; do
  sha=$(sha256sum build-next/$program | cut -c1-16)
  install -m 755 build-next/$program bin/$program-$sha
  echo "$program $ROOT/code/bin/$program-$sha" >> $manifest.tmp
done
mv $manifest.tmp $manifest
while read -r program path; do
  ln -sfn $(basename $path) bin/$program
  echo "published $path"
done < $manifest
