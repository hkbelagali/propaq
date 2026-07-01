"""
Plot total expectation value (including ECORE) vs coefficient cutoff.
Loads FeS_LUCJ_o*.npz files from results/, groups by coeff_cutoff,
sums values across all shards/order-groups per cutoff, and adds ECORE.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parents[2] / "presentation.mplstyle")
import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

cache  = np.load(SCRIPT_DIR / "hamiltonian_cache.npz", allow_pickle=False)
paulis = cache["paulis"].astype(str)
coeffs = cache["coeffs"].real
pw     = np.array([sum(ch != "I" for ch in s) for s in paulis])
ecore  = float(coeffs[pw == 0].sum())
print(f"ECORE: {ecore:.10f}")

# group partial EVs by cutoff value
cutoff_evs: dict[float, float] = defaultdict(float)

for path in sorted(RESULTS_DIR.glob("FeS_LUCJ_o*.npz")):
    d = np.load(path, allow_pickle=False)
    if "values" not in d or "coeff_cutoff" not in d:
        print(f"Skipping {path.name}: missing required keys")
        continue
    cutoff = float(d["coeff_cutoff"])
    cutoff_evs[cutoff] += float(np.asarray(d["values"]).sum())
    print(f"  {path.name}  cutoff={cutoff:.1e}  partial_ev={float(np.asarray(d['values']).sum()):.6f}")

if not cutoff_evs:
    raise FileNotFoundError(f"No valid FeS_LUCJ_o*.npz files found in {RESULTS_DIR}")

cutoffs   = np.array(sorted(cutoff_evs))
total_evs = np.array([ecore + cutoff_evs[c] for c in cutoffs])

print(f"\nCoeff cutoffs: {cutoffs}")
print(f"Total EVs:     {total_evs}")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(cutoffs, total_evs, "s-", lw=2, ms=7, color="black")

legend_entries = [
    plt.Line2D([], [], marker="s", color="black", ms=6, lw=0,
               label=f"$\\epsilon$={eps:.0e}:  {val:.4f}")
    for eps, val in zip(cutoffs, total_evs)
]
ax.legend(handles=legend_entries, fontsize=8, title="Total EV per cutoff", loc="best")

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("Coefficient cutoff")
ax.set_ylabel("Expectation value")
ax.set_title("FeS UCJ (2 layer): total EV vs coefficient cutoff")
fig.tight_layout()

out = SCRIPT_DIR / "plot_zce_cutoff.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved {out}")
