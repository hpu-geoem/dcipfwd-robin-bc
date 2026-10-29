#!/bin/bash
#SBATCH -J dcip-list
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --exclusive
#SBATCH --mem-per-cpu=2700
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Run the lines of a run list in order on one exclusive node with 256 cores (submit_all.sh writes the lists with
# scripts/paper_runs.py); each line is "output mesh properties survey ranks options..." relative to the project root:
#   sbatch --export=ALL,LIST=lists/<group>.txt run_list.slurm
# Finished runs (with a .json report) are skipped. The job fails if any run failed.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/env.sh"
set -u
[[ -x "$BINARY" ]] || { echo "no published binary: $BINARY" >&2; exit 1; }
sha=$(sha256sum $BINARY | cut -d' ' -f1)
list=$(readlink -f $LIST)
failed=0
cd $ROOT
while read -r output mesh properties survey ranks options; do
  [[ -z "$output" || "$output" == \#* ]] && continue
  [[ -f $output.json ]] && continue
  mkdir -p $(dirname $output)
  command="$BINARY -m $mesh -p $properties -s $survey -o $output $options"
  {
    echo "# binary sha256 $sha"
    echo "# job $SLURM_JOB_ID node $SLURM_JOB_NODELIST ranks $ranks"
    echo "# command $command"
    /usr/bin/time -v mpirun -np $ranks --map-by core --bind-to core $command < /dev/null
  } > $output.log 2>&1 || { echo "failed: $output"; failed=1; }
done < "$list"
exit $failed
