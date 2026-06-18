#!/bin/bash
# Submit array jobs for every coefficient order of magnitude present in the
# Hamiltonian, filling up to TOTAL_BUDGET total array tasks.
#
# Usage:
#   bash submit_all.sh              # uses defaults below
#   TOTAL_BUDGET=6000 bash submit_all.sh
#
# Strategy: largest orders (most terms) get scavenger queue (cap 5000);
# overflow or small orders use general queue (cap 1000).  Jobs are submitted
# from largest term-count to smallest so the busiest buckets run first.

TOTAL_BUDGET=${TOTAL_BUDGET:-6000}
SCAVENGER_CAP=5000
GENERAL_CAP=1000

cd "$(dirname "$0")"

module purge
module load Miniforge3
conda activate envThe su

mkdir -p logs results

# ── Compute term counts per order ────────────────────────────────────────────
ORDER_COUNTS=$(python3 - <<'PYEOF'
import numpy as np, math, sys
from collections import Counter

try:
    data = np.load("compiled_hamiltonian_cache.npz", allow_pickle=False)
except FileNotFoundError:
    print("compiled_hamiltonian_cache.npz not found — run python build_LUCJ.py first", file=sys.stderr)
    sys.exit(1)

coeffs = data["coeffs"].real
orders = Counter()
for c in coeffs:
    if abs(c) > 0:
        orders[math.floor(math.log10(abs(c)))] += 1

# Print "order n_terms" sorted by order descending (highest coefficients first)
for o, n in sorted(orders.items(), key=lambda x: -x[0]):
    print(f"{o} {n}")
PYEOF
) || exit 1

echo "Order  Coeff range           Terms  Queue       Tasks  Submitted"
echo "-----  --------------------  -----  ----------  -----  ---------"

budget_remaining=$TOTAL_BUDGET

while IFS=" " read -r order n_terms; do
    if [ "$budget_remaining" -le 0 ]; then
        printf "%5d  [1e%+d, 1e%+d)  %6d  --          --     SKIPPED (budget exhausted)\n" \
            "$order" "$order" "$((order+1))" "$n_terms"
        continue
    fi

    # Pick queue: scavenger if budget allows ≥ scavenger cap, else general
    if [ "$budget_remaining" -ge "$SCAVENGER_CAP" ] && [ "$n_terms" -ge 100 ]; then
        n_tasks=$(( n_terms < SCAVENGER_CAP ? n_terms : SCAVENGER_CAP ))
        run_script="run.sh"
        queue="scavenger"
    else
        avail=$(( budget_remaining < GENERAL_CAP ? budget_remaining : GENERAL_CAP ))
        n_tasks=$(( n_terms < avail ? n_terms : avail ))
        run_script="run_general.sh"
        queue="general"
    fi

    sbatch \
        --job-name="FeS-LUCJ-o${order}" \
        --array=0-$(( n_tasks - 1 )) \
        --export=ALL,ORDER=${order},N_TASKS=${n_tasks} \
        "$run_script" > /dev/null

    printf "%5d  [1e%+d, 1e%+d)  %6d  %-10s  %5d  submitted\n" \
        "$order" "$order" "$((order+1))" "$n_terms" "$queue" "$n_tasks"

    budget_remaining=$(( budget_remaining - n_tasks ))
done <<< "$ORDER_COUNTS"

echo ""
echo "Tasks submitted: $(( TOTAL_BUDGET - budget_remaining )) / ${TOTAL_BUDGET}"
