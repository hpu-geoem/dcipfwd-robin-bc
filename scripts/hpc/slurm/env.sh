# Environment of the job scripts, sourced from the directory of this file. Site settings (modules that provide MPI,
# MFEM and Python; SBATCH_PARTITION and other sbatch defaults) go in site.sh next to this file.
[[ -f "$SLURM_SUBMIT_DIR/site.sh" ]] && source "$SLURM_SUBMIT_DIR/site.sh"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ROOT=$(cd "$SLURM_SUBMIT_DIR/../../.." && pwd)
# A batch pinned to a build (RELEASE=code/bin/release-<build job>) uses its programs; otherwise the published links,
# resolved once per job.
if [[ -n "${RELEASE:-}" ]]; then
  export BINARY=$(awk '$1 == "dcipfwd" {print $2}' $RELEASE) OCTREE_MESH=$(awk '$1 == "octree_mesh" {print $2}' $RELEASE)
else
  export BINARY=$(readlink -f $ROOT/code/bin/dcipfwd 2>/dev/null || true)
  export OCTREE_MESH=$(readlink -f $ROOT/code/bin/octree_mesh 2>/dev/null || true)
fi
echo "date=$(date -Is) job=${SLURM_JOB_ID:-} binary=$BINARY release=${RELEASE:-}"
