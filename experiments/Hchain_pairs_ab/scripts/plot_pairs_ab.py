"""
plot_pairs_ab.py — Plot expectation value and runtime vs pairs_ab configuration.

Reads results directly (no gather step required).

Usage (run from experiments/Hchain_pairs_ab/):
    python scripts/plot_pairs_ab.py

Outputs:
    plots/pairs_ab_energy.pdf
    plots/pairs_ab_runtime.pdf
"""

import re
from collections import defaultdict
from math import floor, log10
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(Path(__file__).resolve().parents[2] / "presentation.mplstyle")

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONNECTIVITIES  = ["square", "heavy-hex"]
ORBITAL_CUTOFFS = [16, 24, 32]
SPACINGS        = [4, 2, 1]
NATOMS          = 30
ORDERS          = [-1, -2]

CONNECTIVITY_COLOR  = {"heavy-hex": "tab:blue", "square": "tab:orange"}
CONNECTIVITY_LABEL  = {"heavy-hex": "heavy-hex", "square": "square"}
CUTOFF_STYLE        = {16: "-", 24: "--", 32: ":"}
CUTOFF_MARKER       = {16: "o",  24: "s",  32: "^"}
SPACING_LABEL       = {4: "spacing=4", 2: "spacing=2", 1: "spacing=1"}

FILE_RE = re.compile(
    r"Hchain_n(\d+)_([\w-]+)_c(\d+)_s(\d+)_o(-?\d+)_(\d+)of(\d+)\.npz$"
)

# ── Load results ──────────────────────────────────────────────────────────────
# ev[(connectivity, cutoff, spacing)] = total EV (ECORE + o-1 + o-2)
# rt[(connectivity, cutoff, spacing)] = total runtime (seconds, summed over orders)

ev_by_order = defaultdict(float)   # (connectivity, cutoff, spacing, order) -> float
rt_by_order = defaultdict(float)

for path in RESULTS_DIR.glob("*.npz"):
    m = FILE_RE.match(path.name)
    if not m:
        continue
    natoms, connectivity, cutoff, spacing, order = (
        int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
    )
    if natoms != NATOMS or order not in ORDERS:
        continue
    d = np.load(path, allow_pickle=False)
    key = (connectivity, cutoff, spacing, order)
    ev_by_order[key] += float(np.sum(d["values"]))
    if "runtime_seconds" in d:
        rt_by_order[key] += float(np.sum(d["runtime_seconds"]))

# Aggregate over orders and add ECORE
ev_total = {}  # (connectivity, cutoff, spacing) -> float
rt_total = {}

for connectivity in CONNECTIVITIES:
    for cutoff in ORBITAL_CUTOFFS:
        for spacing in SPACINGS:
            key = (connectivity, cutoff, spacing)

            cache_path = BASE_DIR / f"n{NATOMS}" / connectivity / f"c{cutoff}_s{spacing}" / "hamiltonian_cache.npz"
            ecore = 0.0
            e_ccsd = None
            if cache_path.exists():
                c      = np.load(cache_path, allow_pickle=False)
                coeffs = c["coeffs"].real
                paulis = c["paulis"].astype(str)
                pw     = np.array([sum(ch != "I" for ch in p) for p in paulis])
                ecore  = float(coeffs[pw == 0].sum())
                if "e_ccsd" in c:
                    e_ccsd = float(c["e_ccsd"])

            ev = ecore + sum(ev_by_order.get((connectivity, cutoff, spacing, o), 0.0) for o in ORDERS)
            rt = sum(rt_by_order.get((connectivity, cutoff, spacing, o), float("nan")) for o in ORDERS)

            if any((connectivity, cutoff, spacing, o) in ev_by_order for o in ORDERS):
                ev_total[key] = ev
                rt_total[key] = rt
                if e_ccsd is not None:
                    ev_total[(connectivity, "ccsd", spacing)] = e_ccsd  # store once per connectivity

# Grab CCSD reference (same for all configs since it's the same molecule)
ccsd_ref = next(
    (float(np.load(BASE_DIR / f"n{NATOMS}" / c / f"c{ORBITAL_CUTOFFS[0]}_s{SPACINGS[0]}" / "hamiltonian_cache.npz")["e_ccsd"])
     for c in CONNECTIVITIES
     if (BASE_DIR / f"n{NATOMS}" / c / f"c{ORBITAL_CUTOFFS[0]}_s{SPACINGS[0]}" / "hamiltonian_cache.npz").exists()
     and "e_ccsd" in np.load(BASE_DIR / f"n{NATOMS}" / c / f"c{ORBITAL_CUTOFFS[0]}_s{SPACINGS[0]}" / "hamiltonian_cache.npz")),
    None
)

# ── Plot ──────────────────────────────────────────────────────────────────────

x_ticks   = list(range(len(SPACINGS)))
x_labels  = [str(s) for s in SPACINGS]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
fig.suptitle(f"H{NATOMS} chain — pairs_ab parameter sweep")

for col, connectivity in enumerate(CONNECTIVITIES):
    ax_ev = axes[0][col]
    ax_rt = axes[1][col]
    color = CONNECTIVITY_COLOR[connectivity]

    for cutoff in ORBITAL_CUTOFFS:
        ys_ev, ys_rt = [], []
        for spacing in SPACINGS:
            key = (connectivity, cutoff, spacing)
            ys_ev.append(ev_total.get(key, float("nan")))
            ys_rt.append(rt_total.get(key, float("nan")) / 3600)

        label = f"cutoff={cutoff}"
        ax_ev.plot(x_ticks, ys_ev,
                   color=color, linestyle=CUTOFF_STYLE[cutoff],
                   marker=CUTOFF_MARKER[cutoff], label=label)
        ax_rt.plot(x_ticks, ys_rt,
                   color=color, linestyle=CUTOFF_STYLE[cutoff],
                   marker=CUTOFF_MARKER[cutoff], label=label)

    if ccsd_ref is not None:
        ax_ev.axhline(ccsd_ref, color="black", linestyle=":", linewidth=1, label="CCSD")

    ax_ev.set_title(connectivity)
    ax_ev.set_ylabel("Energy (Ha)")
    ax_ev.legend(loc='best', fontsize=8)

    ax_rt.set_ylabel("Runtime (hours)")
    ax_rt.set_xlabel("pairs_ab spacing")
    ax_rt.set_xticks(x_ticks)
    ax_rt.set_xticklabels(x_labels)
    ax_rt.legend(loc='best', fontsize=8)

axes[0][0].set_ylabel("Energy (Ha)")
axes[1][0].set_ylabel("Runtime (hours)")

fig.tight_layout()
out = PLOTS_DIR / "pairs_ab_energy_runtime.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
