#!/bin/bash
# Build and test the code on a Slurm cluster, then run the examples of the paper, all pinned to the new build:
#   cd scripts/hpc/slurm && ./submit_all.sh [build job id]        (a given id reuses a finished build)
# Site settings (modules, SBATCH_PARTITION) go in site.sh next to this file.
# MESHES=1 also regenerates the octree meshes of examples 3 and 4 (otherwise they must exist). Each job group of
# scripts/paper_runs.py (five groups of the examples and the ip-coefficients group) runs on its own exclusive node, so
# that the times within a group come from one node.
set -eo pipefail
cd "$(dirname "$0")"
[[ -f site.sh ]] && source site.sh
ROOT=$(cd ../../.. && pwd)
mkdir -p logs lists
build=${1:-$(sbatch --parsable build.slurm)}
export RELEASE=$ROOT/code/bin/release-$build
dependency=afterok:$build
if [[ "${MESHES:-0}" == 1 ]]; then
  mesh=$(sbatch --parsable --dependency=$dependency --export=ALL octree_mesh.slurm)
  dependency=afterok:$mesh
fi
echo "build $build, release $RELEASE"
for group in example1 example2 example3 example4-main example4-aux ip-coefficients; do
  python3 -I ../../paper_runs.py list $group > lists/$group.txt
  sbatch --dependency=$dependency -J $group --export=ALL,LIST=lists/$group.txt run_list.slurm
done
squeue -u $USER -o "%.10i %.14j %.9T %.10M %.5D %R"
