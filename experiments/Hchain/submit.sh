#!/bin/bash
# ── Change these to match your build_LUCJ.py run ─────────────────────────────
NATOMS=20
CONNECTIVITY="heavy-hex"
ORDER=-2
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

module purge
module load Miniforge3
conda activate env

# Count terms in the requested order-of-magnitude bucket from the Hamiltonian cache
N_TERMS=$(python3 - <<PYEOF
import sys, numpy as np
from math import floor, log10

cache = "n${NATOMS}/${CONNECTIVITY}/hamiltonian_cache.npz"
try:
    data = np.load(cache, allow_pickle=False)
except FileNotFoundError:
    print(f"Cache not found — run: python build_LUCJ.py --natoms ${NATOMS} --connectivity ${CONNECTIVITY}", file=sys.stderr)
    sys.exit(1)

coeffs = data["coeffs"].real
n = sum(abs(c) > 0 and floor(log10(abs(c))) == $ORDER for c in coeffs)
print(n)
PYEOF
) || exit 1

if [ "$N_TERMS" -eq 0 ]; then
    echo "No order-${ORDER} terms found. Nothing to submit."
    exit 0
fi

N_TASKS=$(( N_TERMS < 4000 ? N_TERMS : 4000 ))
echo "H${NATOMS} (${CONNECTIVITY}) order-${ORDER}: ${N_TERMS} terms → ${N_TASKS} array tasks"

mkdir -p logs results

sbatch \
    --job-name="Hchain-n${NATOMS}-o${ORDER}" \
    --array=0-$(( N_TASKS - 1 )) \
    --export=ALL,NATOMS=${NATOMS},CONNECTIVITY=${CONNECTIVITY},ORDER=${ORDER} \
    run.sh
