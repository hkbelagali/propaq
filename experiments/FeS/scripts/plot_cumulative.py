"""
plot_cumulative.py — Compare cumulative EV and runtime for noisy vs noiseless runs,
                     and plot cumulative EV / runtime for noiseless-coeff (by order).

Usage (run from anywhere):
    python plot_cumulative.py --system {full-ansatz,noiseless,noiseless-coeff}
                              [--hamiltonian-cache PATH]
                              [--noisy-dir        PATH]
                              [--noiseless-dir    PATH]
                              [--coeff-dir        PATH]
                              [--out-ev           PATH]
                              [--out-rt           PATH]
                              [--out-ev-coeff     PATH]
                              [--out-rt-coeff     PATH]

--system selects which system's refined_data is the primary (noisy) dataset.
The noiseless and noiseless-coeff dirs default to those sibling directories.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

IBM_TRUE_VALUE = -326.7
NPZ_RE_W = re.compile(r"FeS_LUCJ_w(\d+)\.npz$")
NPZ_RE_O = re.compile(r"FeS_LUCJ_o(-?\d+)\.npz$")

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=["full-ansatz", "noiseless", "noiseless-coeff"], required=True,
                help="Primary (noisy) system; noiseless and noiseless-coeff are always the baselines")
ap.add_argument("--hamiltonian-cache", default=None)
ap.add_argument("--noisy-dir",         default=None)
ap.add_argument("--noiseless-dir",     default=None)
ap.add_argument("--coeff-dir",         default=None)
ap.add_argument("--out-ev",            default=None)
ap.add_argument("--out-rt",            default=None)
ap.add_argument("--out-ev-coeff",      default=None)
ap.add_argument("--out-rt-coeff",      default=None)
args = ap.parse_args()

system_dir = BASE_DIR / args.system
plots_dir  = BASE_DIR / "plots"

ham_path    = Path(args.hamiltonian_cache) if args.hamiltonian_cache else BASE_DIR / "hamiltonian_cache.npz"
noisy_dir   = Path(args.noisy_dir)         if args.noisy_dir         else system_dir / "refined_data"
nl_dir      = Path(args.noiseless_dir)     if args.noiseless_dir     else BASE_DIR / "noiseless" / "refined_data"
coeff_dir   = Path(args.coeff_dir)         if args.coeff_dir         else BASE_DIR / "noiseless-coeff" / "refined_data"
out_ev      = Path(args.out_ev)            if args.out_ev            else plots_dir / "cumulative_ev.png"
out_rt      = Path(args.out_rt)            if args.out_rt            else plots_dir / "runtime_boxplot.png"
out_ev_c    = Path(args.out_ev_coeff)      if args.out_ev_coeff      else plots_dir / "cumulative_ev_coeff.png"
out_rt_c    = Path(args.out_rt_coeff)      if args.out_rt_coeff      else plots_dir / "runtime_boxplot_coeff.png"

plots_dir.mkdir(parents=True, exist_ok=True)

# ── Load ECORE ────────────────────────────────────────────────────────────────

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

# ── Load refined_data helper ──────────────────────────────────────────────────

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
print(f"Noisy weights ({args.system}): {sorted(noisy_data)}")
print(f"Noiseless weights:             {sorted(noiseless_data)}")
print(f"Coeff orders:                  {sorted(coeff_data)}")

# ── Build cumulative EV series ────────────────────────────────────────────────

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

# ── Plot 1: Cumulative EV ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(noisy_x, noisy_y, marker="o", lw=1.8, color="steelblue",
        label=f"Noisy ({args.system})", zorder=3)
ax.plot(noiseless_x, noiseless_y, marker="s", lw=1.8, color="darkorange",
        label="Noiseless", zorder=3)

for xi, yi in zip(noisy_x, noisy_y):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(-6, 7), ha="center", fontsize=7, color="steelblue")
for xi, yi in zip(noiseless_x, noiseless_y):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(6, -13), ha="center", fontsize=7, color="darkorange")

ax.axhline(IBM_TRUE_VALUE, color="tomato", lw=1.5, ls="--",
           label=f"IBM true value ({IBM_TRUE_VALUE})")

all_x = sorted(set(noisy_x) | set(noiseless_x))
ax.set_xticks(all_x)
ax.set_xticklabels([("ECORE" if x == 0 else f"w{x}") for x in all_x], fontsize=9)
ax.set_xlabel("Pauli weight included through this point")
ax.set_ylabel("Cumulative $\\langle H \\rangle$")
ax.set_title("FeS LUCJ — Cumulative expectation value by Pauli weight")
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_ev, dpi=150, bbox_inches="tight")
print(f"Saved {out_ev}")
plt.close(fig)

# ── Plot 2: Runtime box plot ──────────────────────────────────────────────────

all_weights = sorted(set(noisy_data) | set(noiseless_data))
width = 0.35  # offset for side-by-side boxes

fig, ax = plt.subplots(figsize=(11, 5))

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

# Legend proxies
from matplotlib.patches import Patch
handles = []
if bp_noisy:
    handles.append(Patch(facecolor=(*plt.cm.tab10(0)[:3], 0.55), label=f"Noisy ({args.system})"))
if bp_nl:
    handles.append(Patch(facecolor=(*plt.cm.tab10(1)[:3], 0.55), label="Noiseless"))
ax.legend(handles=handles, fontsize=9)

ax.set_xticks(range(1, len(all_weights) + 1))
ax.set_xticklabels([f"w{w}" for w in all_weights], fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Pauli weight")
ax.set_ylabel("Per-term runtime (s)")
ax.set_title("FeS LUCJ — Per-term wall-clock runtime by Pauli weight")
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_rt, dpi=150, bbox_inches="tight")
print(f"Saved {out_rt}")
plt.close(fig)

# ── Plot 3: Cumulative EV by coefficient order ────────────────────────────────

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
    f"o{o}\n$[10^{{{o}}},10^{{{o+1}}})$" for o in sorted_orders
]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x_pos_c, coeff_y, marker="D", lw=1.8, color="mediumseagreen", zorder=3,
        label="Noiseless-coeff cumulative EV")
for xi, yi in zip(x_pos_c, coeff_y):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7.5, color="mediumseagreen")

ax.axhline(IBM_TRUE_VALUE, color="tomato", lw=1.5, ls="--",
           label=f"IBM true value ({IBM_TRUE_VALUE})")

ax.set_xticks(x_pos_c)
ax.set_xticklabels(x_labels_c, fontsize=9)
ax.set_xlabel("Coefficient order included through this point")
ax.set_ylabel("Cumulative $\\langle H \\rangle$")
ax.set_title("FeS LUCJ (noiseless-coeff) — Cumulative EV by coefficient order")
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_ev_c, dpi=150, bbox_inches="tight")
print(f"Saved {out_ev_c}")
plt.close(fig)

# ── Plot 4: Runtime box plot by coefficient order ─────────────────────────────

rt_orders, rt_boxes_c = [], []
for o in sorted_orders:
    valid = coeff_data[o]["runtimes"]
    valid = valid[~np.isnan(valid)]
    if valid.size > 0:
        rt_orders.append(o)
        rt_boxes_c.append(valid)

fig, ax = plt.subplots(figsize=(8, 5))

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
        [f"o{o}\n$[10^{{{o}}},10^{{{o+1}}})$\n(n={len(b)})"
         for o, b in zip(rt_orders, rt_boxes_c)],
        fontsize=8,
    )

ax.set_yscale("log")
ax.set_xlabel("Coefficient order")
ax.set_ylabel("Per-term runtime (s)")
ax.set_title("FeS LUCJ (noiseless-coeff) — Per-term runtime by coefficient order")
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(out_rt_c, dpi=150, bbox_inches="tight")
print(f"Saved {out_rt_c}")
plt.close(fig)
