"""
plot_truncation_l1.py — Box plot of total discarded_coeff_l1 by Pauli weight.

Each data point is one JSONL file (one monomial run).
The value plotted is the sum of discarded_coeff_l1 across all truncation events in that file.
Files with no truncation events are skipped.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from propaq import LogParser

RESULTS_DIR = Path(__file__).parent / "results"

weight_data: dict[int, list[float]] = defaultdict(list)

for path in sorted(RESULTS_DIR.glob("*.jsonl")):
    m = re.search(r"_w(\d+)_", path.name)
    if not m:
        continue
    weight = int(m.group(1))

    try:
        parser = LogParser(str(path))
    except (OSError, ValueError):
        continue

    l1_values = parser.discarded_coeff_l1
    if l1_values:
        weight_data[weight].append(sum(l1_values))

weights = sorted(weight_data)
data = [weight_data[w] for w in weights]
labels = [f"w{w}" for w in weights]

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(data, labels=labels, sym=".", medianprops={"color": "C1"})
ax.set_yscale("log")
ax.set_xlabel("Pauli weight")
ax.set_ylabel("Total discarded $\\ell_1$ norm")
ax.set_title("Total coefficient weight dropped per monomial run by observable weight\n(FeS LUCJ)")
ax.set_xticklabels([f"{lbl}\n(n={len(d)})" for lbl, d in zip(labels, data)])

out = Path(__file__).parent / "truncation_l1_by_weight.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
