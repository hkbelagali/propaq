"""
Plot total expectation value (including ECORE) vs coefficient cutoff.
Loads FeS_LUCJ_zce_o*.npz files from results/, sums summed_coeff_values
across orders and adds ECORE from the Hamiltonian cache for each cutoff.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parents[2] / "presentation.mplstyle")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
ZCE_RE = re.compile(r"FeS_LUCJ_zce_o(-?\d+)_\d+of\d+\.npz$")

cache = np.load(SCRIPT_DIR / "compiled_hamiltonian_cache.npz", allow_pickle=False)
paulis = cache["paulis"].astype(str)
coeffs = cache["coeffs"].real
pw = np.array([sum(ch != "I" for ch in s) for s in paulis])
ecore = float(coeffs[pw == 0].sum())
print(f"ECORE: {ecore:.10f}")

cutoff_values = None
order_evs: dict[int, np.ndarray] = {}

for path in sorted(RESULTS_DIR.glob("FeS_LUCJ_zce_o*.npz")):
    m = ZCE_RE.match(path.name)
    if not m:
        continue
    order = int(m.group(1))
    d = np.load(path, allow_pickle=False)
    if "summed_coeff_values" not in d or "coeff_cutoff_values" not in d:
        print(f"Skipping {path.name}: missing required keys")
        continue
    cv = np.asarray(d["coeff_cutoff_values"], dtype=float)
    sv = np.asarray(d["summed_coeff_values"], dtype=float)
    if cutoff_values is None:
        cutoff_values = cv
    elif not np.allclose(cutoff_values, cv):
        print(f"WARNING: {path.name} has different cutoff_values — skipping")
        continue
    order_evs[order] = sv
    print(f"  order {order:>3}: {sv}")

if not order_evs:
    raise FileNotFoundError(f"No valid ZCE npz files found in {RESULTS_DIR}")

total_ev = ecore + sum(order_evs.values())

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(cutoff_values, total_ev, "s-", lw=2, ms=7, color="black")

legend_entries = [
    plt.Line2D([], [], marker="s", color="black", ms=6, lw=0,
               label=f"$\\epsilon$={eps:.0e}:  {val:.4f}")
    for eps, val in zip(cutoff_values, total_ev)
]
ax.legend(handles=legend_entries, fontsize=8, title="Total EV per cutoff", loc='best')

ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("Coefficient cutoff")
ax.set_ylabel("Expectation value")
ax.set_title("FeS LUCJ (3-layer noiseless): total EV vs coefficient cutoff")
fig.tight_layout()

out = SCRIPT_DIR / "plot_zce_cutoff.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved {out}")

print(f"\nCoeff cutoffs:  {cutoff_values}")
print(f"Total EV:       {total_ev}")
