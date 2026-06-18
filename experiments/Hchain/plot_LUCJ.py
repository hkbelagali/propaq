"""
plot_LUCJ.py — Plot LUCJ order-(-1) expectation values and CCSD reference energies
               vs system size, and average per-term runtime vs system size.

Run from examples/Hchain/ after gather.py has populated refined_data/.

Outputs:
    Hchain_energy_vs_natoms.pdf
    Hchain_runtime_vs_natoms.pdf
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 12})

REFINED_DIR = Path("refined_data")
TARGET_ORDER = -1

# Only match canonical files (no _c6/_c10 suffix variants)
FILE_RE = re.compile(r"^Hchain_n(\d+)_([\w-]+)_o(-?\d+)\.npz$")

CONNECTIVITY_STYLE = {
    "heavy-hex": ("tab:blue",   "o", "-"),
    "square":    ("tab:orange", "s", "--"),
    "all-to-all":("tab:green",  "^", "-."),
}

# ── Load refined_data files ───────────────────────────────────────────────────

data = {}  # (natoms, connectivity) -> {"ev_sum": float, "rt_mean": float}

for path in REFINED_DIR.glob("Hchain_*.npz"):
    m = FILE_RE.match(path.name)
    if m is None:
        continue
    natoms, connectivity, order = int(m.group(1)), m.group(2), int(m.group(3))
    if order != TARGET_ORDER:
        continue
    d = np.load(path, allow_pickle=False)
    data[(natoms, connectivity)] = {
        "ev_sum":  float(d["ev_sum"]),
        "rt_mean": float(d["rt_mean"]) if "rt_mean" in d else float("nan"),
    }

if not data:
    raise FileNotFoundError(
        f"No refined_data files found for order {TARGET_ORDER}. Run gather.py first."
    )

# ── Load CCSD energies from hamiltonian caches ────────────────────────────────

ccsd = {}  # natoms -> float

for natoms, connectivity in data:
    if natoms in ccsd:
        continue
    cache_path = Path(f"n{natoms}/{connectivity}/hamiltonian_cache.npz")
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=False)
        if "e_ccsd" in c:
            ccsd[natoms] = float(c["e_ccsd"])

connectivities = sorted({c for _, c in data})
natoms_all = sorted({n for n, _ in data})

# ── Plot 1: Energy vs natoms ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(8, 5))

for connectivity in connectivities:
    color, marker, ls = CONNECTIVITY_STYLE.get(connectivity, ("gray", "o", "-"))
    xs = sorted(n for n, c in data if c == connectivity)
    ys = [data[(n, connectivity)]["ev_sum"] for n in xs]
    ax1.plot(xs, ys, color=color, marker=marker, linestyle=ls,
             label=f"LUCJ ({connectivity})")

if ccsd:
    xs_ref = sorted(ccsd)
    ys_ref = [ccsd[n] for n in xs_ref]
    ax1.plot(xs_ref, ys_ref, color="black", marker="D", linestyle=":",
             label="CCSD")

ax1.set_xlabel("Number of H atoms")
ax1.set_ylabel("Energy (Ha)")
ax1.set_title(f"H chain: order-{TARGET_ORDER} LUCJ expectation value vs CCSD")
ax1.legend()
fig1.tight_layout()
fig1.savefig("Hchain_energy_vs_natoms.png")
print("Saved Hchain_energy_vs_natoms.png")

# ── Plot 2: Average runtime per term vs natoms ────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))

for connectivity in connectivities:
    color, marker, ls = CONNECTIVITY_STYLE.get(connectivity, ("gray", "o", "-"))
    xs = sorted(n for n, c in data if c == connectivity)
    ys = [data[(n, connectivity)]["rt_mean"] for n in xs]
    ax2.plot(xs, ys, color=color, marker=marker, linestyle=ls, label=connectivity)

ax2.set_xlabel("Number of H atoms")
ax2.set_ylabel("Mean runtime per term (s)")
ax2.set_title("H chain: average propagation runtime per Majorana term")
ax2.legend()
fig2.tight_layout()
fig2.savefig("Hchain_runtime_vs_natoms.png")
print("Saved Hchain_runtime_vs_natoms.png")

plt.show()
