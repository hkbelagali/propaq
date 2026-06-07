#!/bin/bash
# ── Change this to run a different Pauli weight ──────────────────────────────
WEIGHT=1
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

module purge
module load Miniforge3
conda activate env

# Count weight-N Pauli terms from the physical Hamiltonian cache (built by build_LUCJ.py)
N_TERMS=$(python3 - <<PYEOF
import sys, numpy as np

cache = "compiled_hamiltonian_cache.npz"
try:
    data = np.load(cache, allow_pickle=False)
except FileNotFoundError:
    print(f"compiled_hamiltonian_cache.npz not found — run python build_LUCJ.py first", file=sys.stderr)
    sys.exit(1)

paulis = data["paulis"].astype(str)
n = sum(sum(c != "I" for c in lbl) == $WEIGHT for lbl in paulis)
print(n)
PYEOF
) || exit 1

if [ "$N_TERMS" -eq 0 ]; then
    echo "No weight-${WEIGHT} terms found. Nothing to submit."
    exit 0
fi

N_TASKS=$(( N_TERMS < 5000 ? N_TERMS : 5000 ))
echo "Weight-${WEIGHT}: ${N_TERMS} terms → ${N_TASKS} array tasks"

mkdir -p logs results

sbatch \
    --job-name="FeS-LUCJ-w${WEIGHT}" \
    --array=0-$(( N_TASKS - 1 )) \
    --export=ALL,WEIGHT=${WEIGHT} \
    run.sh
