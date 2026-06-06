#!/bin/bash

#SBATCH --job-name=FeS-LUCJ
#SBATCH --partition=scavenger
#SBATCH --account=general
#SBATCH --qos=scavenger
#SBATCH --time=01:30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err

cd "${SLURM_SUBMIT_DIR}"

set -eo pipefail

module purge
module load Miniforge3

conda activate env

mkdir -p logs results

python run_LUCJ.py \
    --weight  "$WEIGHT" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --n-tasks "${N_TASKS:-$SLURM_ARRAY_TASK_COUNT}"
