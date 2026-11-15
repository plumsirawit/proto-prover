#!/usr/bin/env bash
#SBATCH --job-name=latent-proof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:45:00
#SBATCH --open-mode=append

set -euo pipefail
repo=$1
root=$2
python=$3
lean=$4
size=$5
shift 5
module=latent_prover.experiment
if [[ ${1:-} == --structured ]]; then
    module=latent_prover.structured
    shift
fi

memory_mib=$(( SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK ))
if (( SLURM_CPUS_PER_TASK > 8 || memory_mib > 24576 )); then
    printf 'This experiment is limited to 8 CPUs and 24 GiB.\n' >&2
    exit 1
fi

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

run="$root/${size}-${SLURM_JOB_ID}"
mkdir "$run"
cp -R "$repo/latent_prover" "$run/"
cp "$repo/lean-toolchain" "$repo/scripts/experiment.sbatch" "$run/"
cd "$run"
sha256sum latent_prover/*.py latent_prover/*.lean lean-toolchain experiment.sbatch > sources.sha256
printf 'Job %s on %s with %s CPUs and %s MiB\n' "$SLURM_JOB_ID" "$(hostname)" \
    "$SLURM_CPUS_PER_TASK" "$memory_mib"
"$lean" --version
exec srun --cpu-bind=cores "$python" -m "$module" \
    --size "$size" --output "$run/results" --lean "$lean" \
    --threads "$SLURM_CPUS_PER_TASK" --verify-workers "$(( (SLURM_CPUS_PER_TASK + 1) / 2 ))" "$@"
