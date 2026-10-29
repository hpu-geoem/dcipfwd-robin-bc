#!/bin/bash
#SBATCH -J dcip-octree
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=200G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Refine the root meshes of the octree Overthrust models with octree_mesh (submit_all.sh submits it):
#   mkdir -p logs && sbatch octree_mesh.slurm      (CASES="octree-section octree-volume" by default)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"
set -u
OCTREE=$OCTREE_MESH
[[ -x "$OCTREE" ]] || { echo "no published octree_mesh" >&2; exit 1; }
pids=()
for case in ${CASES:-octree-section octree-volume}; do
  directory=$ROOT/models/e8-overthrust/$case
  for root in $directory/root-*.mesh; do
    extent=${root##*/root-}
    extent=${extent%.mesh}
    { echo "# octree_mesh sha256 $(sha256sum $OCTREE | cut -d' ' -f1)"
      $OCTREE -m $root -g $directory/voxels -bg $directory/background.txt -o $directory/model-$extent \
        $(cat $directory/octree-options.txt); } > $directory/model-$extent.log 2>&1 &
    pids+=($!)
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait $pid || failed=1
done
md5sum $ROOT/models/e8-overthrust/octree-*/model-500.cells
exit $failed
