#!/bin/bash
# ── Change this to run a different coefficient order of magnitude ─────────────
ORDER=${ORDER:--2}
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

module purge
module load Miniforge3
conda activate env

# Count terms whose floor(log10|coeff|) == ORDER
N_TERMS=$(python3 - <<PYEOF
import sys, numpy as np, math

cache = "compiled_hamiltonian_cache.npz"
try:
    data = np.load(cache, allow_pickle=False)
except FileNotFoundError:
    print(f"compiled_hamiltonian_cache.npz not found — run python build_LUCJ.py first", file=sys.stderr)
    sys.exit(1)

coeffs = data["coeffs"].real
n = sum(
    abs(c) > 0 and math.floor(math.log10(abs(c))) == $ORDER
    for c in coeffs
)
print(n)
PYEOF
) || exit 1

if [ "$N_TERMS" -eq 0 ]; then
    echo "No order-${ORDER} terms found. Nothing to submit."
    exit 0
fi

N_TASKS=$(( N_TERMS < 1000 ? N_TERMS : 1000 ))
echo "Order-${ORDER}: ${N_TERMS} terms → ${N_TASKS} array tasks"

mkdir -p logs results

sbatch \
    --job-name="FeS-LUCJ-o${ORDER}" \
    --array=0-$(( N_TASKS - 1 )) \
    --export=ALL,ORDER=${ORDER} \
    run_general.sh
