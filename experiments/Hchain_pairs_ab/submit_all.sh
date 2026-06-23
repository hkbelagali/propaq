#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

NATOMS=30
CONNECTIVITIES=("all-to-all")
ORBITAL_CUTOFFS=(16 24 32)
SPACINGS=(4 2 1)

for connectivity in "${CONNECTIVITIES[@]}"; do
    for cutoff in "${ORBITAL_CUTOFFS[@]}"; do
        for spacing in "${SPACINGS[@]}"; do
            echo "=== ${connectivity}  cutoff=${cutoff}  spacing=${spacing} ==="
            for order in -1 -2; do
                echo "--- order=${order} ---"
                python "${DIR}/run_LUCJ.py" \
                    --natoms "${NATOMS}" \
                    --connectivity "${connectivity}" \
                    --orbital-cutoff "${cutoff}" \
                    --spacing "${spacing}" \
                    --order "${order}"
            done
        done
    done
done
