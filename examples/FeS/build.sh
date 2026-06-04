#!/bin/bash

#SBATCH --job-name=FeS-LUCJ-Hamiltonian
#SBATCH --account=general
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "${SLURM_SUBMIT_DIR}"

set -eo pipefail

module purge
module load Miniforge3

conda activate env

python build_hamiltonian.py

