"""
plot_LUCJ.py — Plot LUCJ expectation values (all orders) and CCSD reference
               energies vs system size, and total wall time vs system size.

Usage (run from anywhere):
    python plot_LUCJ.py [--refined-dir PATH] [--out-energy PATH] [--out-runtime PATH]

Outputs (default: experiments/Hchain/plots/):
    Hchain_energy_vs_natoms.pdf
    Hchain_runtime_vs_natoms.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(Path(__file__).resolve().parent.parent.parent / "presentation.mplstyle")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

ap = argparse.ArgumentParser()
ap.add_argument("--refined-dir",  default=None, help="Override refined_data directory")
ap.add_argument("--out-energy",   default=None, help="Override energy plot output path")
ap.add_argument("--out-runtime",  default=None, help="Override runtime plot output path")
args = ap.parse_args()

plots_dir   = BASE_DIR / "plots"
REFINED_DIR = Path(args.refined_dir) if args.refined_dir else BASE_DIR / "refined_data"
out_energy  = Path(args.out_energy)  if args.out_energy  else plots_dir / "Hchain_energy_vs_natoms.pdf"
out_runtime = Path(args.out_runtime) if args.out_runtime else plots_dir / "Hchain_runtime_vs_natoms.pdf"

plots_dir.mkdir(parents=True, exist_ok=True)

# Only match canonical files (no _c6/_c10 suffix variants)
FILE_RE = re.compile(r"^Hchain_n(\d+)_([\w-]+)_o(-?\d+)\.npz$")

CONNECTIVITY_COLOR  = {"heavy-hex": "tab:blue", "square": "tab:orange", "all-to-all": "tab:green"}
CONNECTIVITY_MARKER = {"heavy-hex": "o",        "square": "s",          "all-to-all": "^"}

# ── Load refined_data files ───────────────────────────────────────────────────

data = {}  # (natoms, connectivity, order) -> {"ev_sum": float, "rt_wall": float}

for path in REFINED_DIR.glob("Hchain_*.npz"):
    m = FILE_RE.match(path.name)
    if m is None:
        continue
    natoms, connectivity, order = int(m.group(1)), m.group(2), int(m.group(3))
    d = np.load(path, allow_pickle=False)
    if "runtime_seconds" in d:
        n_tasks = int(d["n_tasks"]) if "n_tasks" in d else 1
        rt_wall = float(np.sum(d["runtime_seconds"])) / n_tasks
    else:
        rt_wall = float("nan")
    data[(natoms, connectivity, order)] = {
        "ev_sum":  float(d["ev_sum"]),
        "rt_wall": rt_wall,
    }

if not data:
    raise FileNotFoundError(
        "No refined_data files found. Run gather.py first."
    )

# Sum EV over all orders; wall time is the max across orders (orders run concurrently)
aggregated = {}  # (natoms, connectivity) -> {"ev_sum": float, "rt_wall": float}
for (natoms, connectivity, order), vals in data.items():
    key = (natoms, connectivity)
    if key not in aggregated:
        aggregated[key] = {"ev_sum": 0.0, "rt_wall": 0.0}
    aggregated[key]["ev_sum"]  += vals["ev_sum"]
    aggregated[key]["rt_wall"] += vals["rt_wall"]

connectivities = sorted({c for _, c in aggregated})
connectivities = [c for c in connectivities if c not in ("square", "all-to-all")]
natoms_all     = sorted({n for n, _ in aggregated})

# ── Load CCSD energies from hamiltonian caches ────────────────────────────────

ccsd = {}  # natoms -> float

for natoms, connectivity in aggregated:
    if natoms in ccsd:
        continue
    cache_path = BASE_DIR / f"n{natoms}" / connectivity / "hamiltonian_cache.npz"
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=False)
        if "e_ccsd" in c:
            ccsd[natoms] = float(c["e_ccsd"])

# ── Plot 1: Energy vs natoms ──────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(6, 4))

for connectivity in connectivities:
    color  = CONNECTIVITY_COLOR.get(connectivity,  "gray")
    marker = CONNECTIVITY_MARKER.get(connectivity, "o")
    xs = sorted(n for n, c in aggregated if c == connectivity)
    ys = [aggregated[(n, connectivity)]["ev_sum"] for n in xs]
    ax1.plot(xs, ys, color=color, marker=marker, linestyle="-",
             label=f"LUCJ ({connectivity})")

if ccsd:
    xs_ref = sorted(ccsd)
    ys_ref = [ccsd[n] for n in xs_ref]
    ax1.plot(xs_ref, ys_ref, color="black", marker="D", linestyle=":",
             label="CCSD")

ax1.set_xlabel("Number of H atoms")
ax1.set_ylabel("Energy (Ha)")
ax1.set_title("H chain: LUCJ expectation value vs CCSD")
ax1.set_xlim(left=0)
ax1.legend()
fig1.tight_layout()
fig1.savefig(out_energy, dpi=150, bbox_inches="tight")
print(f"Saved {out_energy}")

# ── Plot 2: Total wall time vs natoms ─────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(6, 4))

for connectivity in connectivities:
    color  = CONNECTIVITY_COLOR.get(connectivity,  "gray")
    marker = CONNECTIVITY_MARKER.get(connectivity, "o")
    xs = sorted(n for n, c in aggregated if c == connectivity)
    ys = [aggregated[(n, connectivity)]["rt_wall"] / 3600 for n in xs]
    ax2.plot(xs, ys, color=color, marker=marker, linestyle="-",
             label=connectivity)

ax2.set_xlabel("Number of H atoms")
ax2.set_ylabel("Total wall time (hours)")
ax2.set_title("H chain: total wall time for full propagation")
ax2.set_xlim(left=0)
ax2.legend()
fig2.tight_layout()
fig2.savefig(out_runtime, dpi=150, bbox_inches="tight")
print(f"Saved {out_runtime}")

plt.show()
