"""
Compare cumulative EV and runtime for noisy vs noiseless runs, and plot cumulative EV / runtime for noiseless-coeff (by order).
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parent.parent.parent / "presentation.mplstyle")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

LUCJ_PLUS_SQD_ENERGY = -326.7  # From paper.
HARTREE_FOCK_ENERGY = float(np.loadtxt(f"{BASE_DIR}/energy_hf.txt"))
CCSD_ENERGY = float(np.loadtxt(f"{BASE_DIR}/energy_ccsd.txt"))

NPZ_RE_W = re.compile(r"FeS_LUCJ_w(\d+)\.npz$")
NPZ_RE_O = re.compile(r"FeS_LUCJ_o(-?\d+)\.npz$")

ap = argparse.ArgumentParser()
ap.add_argument("--hamiltonian-cache", default=None)
ap.add_argument("--noisy-dir",  default=None)
ap.add_argument("--noiseless-dir", default=None)
ap.add_argument("--coeff-dir", default=None)
ap.add_argument("--out-ev", default=None)
ap.add_argument("--out-rt", default=None)
ap.add_argument("--out-ev-coeff", default=None)
ap.add_argument("--out-rt-coeff", default=None)
args = ap.parse_args()

plots_dir = BASE_DIR / "plots"

ham_path = Path(args.hamiltonian_cache) if args.hamiltonian_cache else BASE_DIR / "hamiltonian_cache.npz"
noisy_dir = Path(args.noisy_dir) if args.noisy_dir else BASE_DIR / "full-ansatz" / "refined_data"
nl_dir = Path(args.noiseless_dir) if args.noiseless_dir else BASE_DIR / "noiseless" / "refined_data"
coeff_dir = Path(args.coeff_dir) if args.coeff_dir else BASE_DIR / "noiseless-coeff" / "refined_data"
out_ev = Path(args.out_ev) if args.out_ev else plots_dir / "cumulative_ev.pdf"
out_rt = Path(args.out_rt) if args.out_rt else plots_dir / "runtime_boxplot.pdf"
out_ev_c = Path(args.out_ev_coeff) if args.out_ev_coeff else plots_dir / "cumulative_ev_coeff.pdf"
out_rt_c = Path(args.out_rt_coeff) if args.out_rt_coeff else plots_dir / "runtime_boxplot_coeff.pdf"

plots_dir.mkdir(parents=True, exist_ok=True)

ecore = None
if ham_path.exists():
    c = np.load(ham_path, allow_pickle=False)
    paulis = c["paulis"].astype(str)
    coeffs = c["coeffs"]
    pw = np.array([sum(ch != "I" for ch in s) for s in paulis])
    ecore = float(coeffs[pw == 0].sum().real)
    print(f"ECORE (weight-0): {ecore:.6f}")
else:
    print("WARNING: hamiltonian_cache.npz not found — ECORE excluded")

EXCLUDE_WEIGHTS = {71, 72}

def _load_entry(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "ev_sum":   float(d["ev_sum"]),
        "runtimes": np.asarray(d["runtime_seconds"], dtype=float).ravel(),
        "missing":  int(d["missing_task_ids"].size),
    }

def load_refined(directory: Path) -> dict[int, dict]:
    data: dict[int, dict] = {}
    for path in directory.glob("FeS_LUCJ_w*.npz"):
        m = NPZ_RE_W.match(path.name)
        if not m:
            continue
        w = int(m.group(1))
        if w in EXCLUDE_WEIGHTS:
            continue
        data[w] = _load_entry(path)
    return data

EXCLUDE_ORDERS = {2}

def load_refined_coeff(directory: Path) -> dict[int, dict]:
    data: dict[int, dict] = {}
    for path in directory.glob("FeS_LUCJ_o*.npz"):
        m = NPZ_RE_O.match(path.name)
        if not m:
            continue
        o = int(m.group(1))
        if o in EXCLUDE_ORDERS:
            continue
        data[o] = _load_entry(path)
    return data

noisy_data     = load_refined(noisy_dir)
noiseless_data = load_refined(nl_dir)
coeff_data     = load_refined_coeff(coeff_dir)
print(f"Noisy weights ({noisy_dir}): {sorted(noisy_data)}")
print(f"Noiseless weights:             {sorted(noiseless_data)}")
print(f"Coeff orders:                  {sorted(coeff_data)}")

def cumulative_series(data: dict[int, dict]) -> tuple[list[int | str], list[float]]:
    """Return (weight_labels, cumulative_ev) starting from ECORE if available."""
    xs, ys = [], []
    cum = ecore if ecore is not None else 0.0
    if ecore is not None:
        xs.append(0)
        ys.append(cum)
    for w in sorted(data):
        cum += data[w]["ev_sum"]
        xs.append(w)
        ys.append(cum)
    return xs, ys

noisy_x,     noisy_y     = cumulative_series(noisy_data)
noiseless_x, noiseless_y = cumulative_series(noiseless_data)

fig, ax = plt.subplots()

ax.plot(noisy_x, noisy_y, marker="o", lw=1.8, color="steelblue",
        label=f"Noisy ($\gamma = 0.001$)", zorder=3)
ax.plot(noiseless_x, noiseless_y, marker="s", lw=1.8, color="darkorange",
        label="Noiseless", zorder=3)

for xi, yi in zip(noisy_x, noisy_y):
    offset = (-10, 0) if xi == 1 else (-6, 8)
    ha     = "right"  if xi == 1 else "center"
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=offset, ha=ha, fontsize=8.5, color="steelblue")
for xi, yi in zip(noiseless_x, noiseless_y):
    if xi == 0:
        offset = (18, -13)   # ECORE: nudge right
    elif xi == 1:
        offset = (6, -13)
    else:
        offset = (6, -26)    # weights 2-11: unified with higher-weight level, plus extra down
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=offset, ha="center", fontsize=8.5, color="darkorange")

ax.axhline(LUCJ_PLUS_SQD_ENERGY, color="black", lw=1.5, ls="--",
           label=f"LUCJ2 + SQD({LUCJ_PLUS_SQD_ENERGY})")
ax.axhline(HARTREE_FOCK_ENERGY, color="tab:blue", lw=1.5, ls="--",
           label=f"Hartree-Fock: ({HARTREE_FOCK_ENERGY})")
ax.axhline(CCSD_ENERGY, color="tab:green", lw=1.5, ls="--",
           label=f"CCSD: ({CCSD_ENERGY})")

all_x = sorted(set(noisy_x) | set(noiseless_x))
ax.set_xticks(all_x)
ax.set_xticklabels([("ECORE" if x == 0 else str(x)) for x in all_x], fontsize=9)
span = all_x[-1] - all_x[0]
ax.set_xlim(all_x[0] - 0.05 * span, all_x[-1] + 0.05 * span)
ax.set_xlabel("Pauli weight")
ax.set_ylabel("Cumulative $\\langle H \\rangle$")
ax.set_title("Cumulative expectation value of FeS LUCJ ansatz by Pauli weight")
ax.set_ylim(top=-270)
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_ev, dpi=150, bbox_inches="tight")
print(f"Saved {out_ev}")
plt.close(fig)

all_weights = sorted(set(noisy_data) | set(noiseless_data))
width = 0.35  # offset for side-by-side boxes

fig, ax = plt.subplots()

noisy_positions, noisy_boxes = [], []
nl_positions,    nl_boxes    = [], []

for i, w in enumerate(all_weights):
    base = i + 1
    if w in noisy_data:
        rt = noisy_data[w]["runtimes"]
        valid = rt[~np.isnan(rt)]
        if valid.size > 0:
            noisy_positions.append(base - width / 2)
            noisy_boxes.append(valid)
    if w in noiseless_data:
        rt = noiseless_data[w]["runtimes"]
        valid = rt[~np.isnan(rt)]
        if valid.size > 0:
            nl_positions.append(base + width / 2)
            nl_boxes.append(valid)

bp_noisy = ax.boxplot(
    noisy_boxes, positions=noisy_positions, widths=width * 0.9, sym=".",
    patch_artist=True,
    medianprops={"color": "white", "lw": 1.5},
    boxprops={"facecolor": (*plt.cm.tab10(0)[:3], 0.55)},
) if noisy_boxes else None

bp_nl = ax.boxplot(
    nl_boxes, positions=nl_positions, widths=width * 0.9, sym=".",
    patch_artist=True,
    medianprops={"color": "white", "lw": 1.5},
    boxprops={"facecolor": (*plt.cm.tab10(1)[:3], 0.55)},
) if nl_boxes else None

from matplotlib.patches import Patch
handles = []
if bp_noisy:
    handles.append(Patch(facecolor=(*plt.cm.tab10(0)[:3], 0.55), label=f"Noisy ($\gamma = 0.001$)"))
if bp_nl:
    handles.append(Patch(facecolor=(*plt.cm.tab10(1)[:3], 0.55), label="Noiseless"))
ax.legend(handles=handles, fontsize=9)

ax.set_xticks(range(1, len(all_weights) + 1))
ax.set_xticklabels([str(w) for w in all_weights], fontsize=9)
ax.set_yscale("log")
ax.margins(y=0.2)
ax.set_xlabel("Pauli weight")
ax.set_ylabel("Per-term runtime (s)")
ax.set_title("Per-term wall-clock runtime of FeS LUCJ ansatz propagation by Pauli weight")
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_rt, dpi=150, bbox_inches="tight")
print(f"Saved {out_rt}")
plt.close(fig)

sorted_orders = sorted(coeff_data)
coeff_x, coeff_y = [], []
cum = ecore if ecore is not None else 0.0
if ecore is not None:
    coeff_x.append(None)   # placeholder; plotted separately
    coeff_y.append(cum)
for o in sorted_orders:
    cum += coeff_data[o]["ev_sum"]
    coeff_x.append(o)
    coeff_y.append(cum)

x_pos_c = list(range(len(coeff_y)))
x_labels_c = (["ECORE"] if ecore is not None else []) + [
    f"$[10^{{{o}}},10^{{{o+1}}})$" for o in sorted_orders
]

fig, ax = plt.subplots()

ax.plot(x_pos_c, coeff_y, marker="D", lw=1.8, color="mediumseagreen", zorder=3,
        label="Propagation")
for xi, yi, order in zip(x_pos_c, coeff_y, coeff_x):
    if order is None:        # ECORE
        offset = (-8, 8)
    elif order in (0, 1):    # [10^0, 10^1) and [10^1, 10^2) buckets
        offset = (10, 8)
    else:
        offset = (0, 8)
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=offset, ha="center", fontsize=7.5, color="mediumseagreen")

ax.axhline(LUCJ_PLUS_SQD_ENERGY, color="black", lw=1.5, ls="--",
           label=f"LUCJ2 + SQD({LUCJ_PLUS_SQD_ENERGY})")
ax.axhline(HARTREE_FOCK_ENERGY, color="tab:blue", lw=1.5, ls="--",
           label=f"Hartree-Fock: ({HARTREE_FOCK_ENERGY})")
ax.axhline(CCSD_ENERGY, color="tab:green", lw=1.5, ls="--",
           label=f"CCSD: ({CCSD_ENERGY})")

ax.set_xticks(x_pos_c)
ax.set_xticklabels(x_labels_c, fontsize=9)
ax.set_xlabel("Coefficient order")
ax.set_ylabel("Cumulative $\\langle H \\rangle$")
ax.set_title("Cumulative Expectation Value by Coefficient Order for noiseless FeS LUCJ ansatz propagation")
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_ev_c, dpi=150, bbox_inches="tight")
print(f"Saved {out_ev_c}")
plt.close(fig)

rt_orders, rt_boxes_c = [], []
for o in sorted_orders:
    valid = coeff_data[o]["runtimes"]
    valid = valid[~np.isnan(valid)]
    if valid.size > 0:
        rt_orders.append(o)
        rt_boxes_c.append(valid)

fig, ax = plt.subplots()

if rt_boxes_c:
    x_pos_r = list(range(1, len(rt_orders) + 1))
    bp = ax.boxplot(rt_boxes_c, positions=x_pos_r, sym=".",
                    patch_artist=True,
                    medianprops={"color": "white", "lw": 1.5})
    cmap = plt.cm.tab10
    for patch, i in zip(bp["boxes"], range(len(rt_orders))):
        patch.set_facecolor((*cmap(i % 10)[:3], 0.55))
    ax.set_xticks(x_pos_r)
    ax.set_xticklabels(
        [f"$[10^{{{o}}},10^{{{o+1}}})$" for o in rt_orders],
        fontsize=8,
    )

ax.set_yscale("log")
ax.margins(y=0.2)
ax.set_xlabel("Coefficient order")
ax.set_ylabel("Per-term runtime (s)")
ax.set_title("Per-term wall-clock runtime of noiseless FeS LUCJ ansatz propagation by coefficient order")
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_rt_c, dpi=150, bbox_inches="tight")
print(f"Saved {out_rt_c}")
plt.close(fig)
