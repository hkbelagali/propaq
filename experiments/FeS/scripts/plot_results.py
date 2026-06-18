"""
plot_results.py — Extract metrics from all JSONL/NPZ files in results/ via LogParser and plot.

Usage:
    python plot_results.py --system {full-ansatz,noiseless,noiseless-coeff}
                           [--results-dir PATH]
                           [--out PATH]

Panels:
    1. Median map_terms growth through the circuit by weight
    2. Median outbox_terms growth through the circuit by weight
    3. Total discarded L1 per task (box by weight)
    4. Max discarded coeff per truncation event (box by weight)
    5. Per-monomial EV distribution (box by weight)
    6. Per-term runtime distribution (box by weight)
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR.parents[1]))

from propaq import LogParser

# ── CLI ───────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=["full-ansatz", "noiseless", "noiseless-coeff"], required=True)
ap.add_argument("--results-dir", default=None, help="Override results directory")
ap.add_argument("--out", default=None, help="Override output plot path")
args = ap.parse_args()

system_dir  = BASE_DIR / args.system
RESULTS_DIR = Path(args.results_dir) if args.results_dir else system_dir / "results"
out_path    = Path(args.out)         if args.out         else system_dir / "plots" / "plot_results.png"

JSONL_RE = re.compile(r"_w(\d+)_(\d+)of(\d+)\.jsonl$")
NPZ_RE   = re.compile(r"_w(\d+)_(\d+)of(\d+)\.npz$")

# ── Load JSONL via LogParser ──────────────────────────────────────────────────

# weight -> list of per-file dicts
jsonl_data: dict[int, list[dict]] = defaultdict(list)

for path in sorted(RESULTS_DIR.glob("*.jsonl")):
    m = JSONL_RE.search(path.name)
    if not m:
        continue
    w = int(m.group(1))
    try:
        lp = LogParser(str(path))
    except (OSError, ValueError):
        continue

    gate_idx   = lp.gate_indices
    map_terms  = lp.map_terms
    outbox     = lp.outbox_terms
    trunc_l1   = lp.discarded_coeff_l1
    trunc_max  = lp.discarded_coeff_max

    if not gate_idx:
        continue

    jsonl_data[w].append({
        "gate_idx":       np.array(gate_idx,  dtype=float),
        "map_terms":      np.array(map_terms,  dtype=float),
        "outbox_terms":   np.array(outbox,     dtype=float),
        "total_l1":       float(sum(trunc_l1)) if trunc_l1 else np.nan,
        "per_event_max":  np.array(trunc_max,  dtype=float),
    })

# ── Load NPZ files ────────────────────────────────────────────────────────────

npz_data: dict[int, dict] = defaultdict(lambda: {"values": [], "runtimes": []})

for path in sorted(RESULTS_DIR.glob("*.npz")):
    m = NPZ_RE.search(path.name)
    if not m:
        continue
    w = int(m.group(1))
    try:
        d = np.load(str(path), allow_pickle=False)
    except Exception:
        continue
    vals = np.asarray(d["values"], dtype=float).ravel()
    npz_data[w]["values"].append(vals)
    if "runtime_seconds" in d:
        rt = np.asarray(d["runtime_seconds"], dtype=float).ravel()
        rt = rt[~np.isnan(rt)]
        npz_data[w]["runtimes"].append(rt)

# ── Helpers ───────────────────────────────────────────────────────────────────

weights = sorted(set(jsonl_data) | set(npz_data))
cmap    = plt.cm.tab10
colors  = {w: cmap(i % 10) for i, w in enumerate(weights)}

N_GRID = 200  # interpolation points for term-count curves


def _interp_median(files: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate per-file series onto a common [0,1] grid and return (x, median, iqr_lo, iqr_hi)."""
    x_grid = np.linspace(0, 1, N_GRID)
    interped = []
    for f in files:
        xi = f["gate_idx"]
        yi = f[key]
        if len(xi) < 2:
            continue
        xn = (xi - xi[0]) / (xi[-1] - xi[0])
        interped.append(np.interp(x_grid, xn, yi))
    if not interped:
        return x_grid, np.full(N_GRID, np.nan), np.full(N_GRID, np.nan), np.full(N_GRID, np.nan)
    mat = np.vstack(interped)
    return (
        x_grid,
        np.median(mat, axis=0),
        np.percentile(mat, 25, axis=0),
        np.percentile(mat, 75, axis=0),
    )


def _box_data(d: dict, key: str) -> list[np.ndarray]:
    """Collect per-weight arrays for a box plot, in sorted-weight order."""
    out = []
    for w in weights:
        if key == "total_l1":
            vals = [f["total_l1"] for f in jsonl_data.get(w, []) if not np.isnan(f["total_l1"])]
            out.append(np.array(vals))
        elif key == "per_event_max":
            arrs = [f["per_event_max"] for f in jsonl_data.get(w, [])]
            out.append(np.concatenate(arrs) if arrs else np.array([]))
        elif key in ("values", "runtimes"):
            arrs = d[w][key] if w in d else []
            out.append(np.concatenate(arrs) if arrs else np.array([]))
        else:
            out.append(np.array([]))
    return out


# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"FeS LUCJ ({args.system}) — Results summary", fontsize=14, y=0.98)

ax_map, ax_out, ax_l1 = axes[0]
ax_max, ax_ev,  ax_rt  = axes[1]

# ── Panel 1 & 2: term count growth ───────────────────────────────────────────

for ax, key, ylabel in (
    (ax_map, "map_terms",    "Local hashmap terms"),
    (ax_out, "outbox_terms", "Outbox terms"),
):
    for w in weights:
        files = jsonl_data.get(w, [])
        if not files:
            continue
        x, med, lo, hi = _interp_median(files, key)
        label = f"w{w} (n={len(files)})"
        ax.plot(x, med, color=colors[w], lw=1.5, label=label)
        ax.fill_between(x, lo, hi, color=colors[w], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Fractional circuit depth")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " vs circuit depth (median ± IQR)")
    ax.legend(fontsize=7, ncol=2)

# ── Panel 3: total discarded L1 per task (box by weight) ─────────────────────

l1_box = _box_data({}, "total_l1")
nonempty = [(i, d) for i, d in enumerate(l1_box) if len(d) > 0]
if nonempty:
    idxs, data = zip(*nonempty)
    bp = ax_l1.boxplot(data, positions=[i + 1 for i in idxs],
                       sym=".", medianprops={"color": "C1"}, patch_artist=True)
    for patch, i in zip(bp["boxes"], idxs):
        patch.set_facecolor((*colors[weights[i]][:3], 0.4))
    ax_l1.set_xticks([i + 1 for i in idxs])
    ax_l1.set_xticklabels([f"w{weights[i]}\n(n={len(l1_box[i])})" for i in idxs], fontsize=8)
ax_l1.set_yscale("log")
ax_l1.set_xlabel("Pauli weight")
ax_l1.set_ylabel("Total discarded $\\ell_1$ (per task)")
ax_l1.set_title("Truncation error — total discarded $\\ell_1$ per task")

# ── Panel 4: per-event discarded max coeff (box by weight) ───────────────────

max_box = _box_data({}, "per_event_max")
nonempty = [(i, d) for i, d in enumerate(max_box) if len(d) > 0]
if nonempty:
    idxs, data = zip(*nonempty)
    bp = ax_max.boxplot(data, positions=[i + 1 for i in idxs],
                        sym=".", medianprops={"color": "C1"}, patch_artist=True)
    for patch, i in zip(bp["boxes"], idxs):
        patch.set_facecolor((*colors[weights[i]][:3], 0.4))
    ax_max.set_xticks([i + 1 for i in idxs])
    ax_max.set_xticklabels([f"w{weights[i]}\n(n={len(max_box[i])})" for i in idxs], fontsize=8)
ax_max.set_yscale("log")
ax_max.set_xlabel("Pauli weight")
ax_max.set_ylabel("Max discarded coeff per event")
ax_max.set_title("Truncation error — max discarded coeff per truncation event")

# ── Panel 5: per-monomial EV distribution (box by weight) ────────────────────

ev_box = _box_data(npz_data, "values")
nonempty = [(i, d) for i, d in enumerate(ev_box) if len(d) > 0]
if nonempty:
    idxs, data = zip(*nonempty)
    bp = ax_ev.boxplot(data, positions=[i + 1 for i in idxs],
                       sym=".", medianprops={"color": "C1"}, patch_artist=True)
    for patch, i in zip(bp["boxes"], idxs):
        patch.set_facecolor((*colors[weights[i]][:3], 0.4))
    ax_ev.set_xticks([i + 1 for i in idxs])
    ax_ev.set_xticklabels([f"w{weights[i]}\n(n={len(ev_box[i])})" for i in idxs], fontsize=8)
ax_ev.axhline(0, color="gray", lw=0.8, ls="--")
ax_ev.set_xlabel("Pauli weight")
ax_ev.set_ylabel("Per-monomial EV")
ax_ev.set_title("Per-monomial expectation value distribution")

# ── Panel 6: per-term runtime distribution (box by weight) ───────────────────

rt_box = _box_data(npz_data, "runtimes")
nonempty = [(i, d) for i, d in enumerate(rt_box) if len(d) > 0]
if nonempty:
    idxs, data = zip(*nonempty)
    bp = ax_rt.boxplot(data, positions=[i + 1 for i in idxs],
                       sym=".", medianprops={"color": "C1"}, patch_artist=True)
    for patch, i in zip(bp["boxes"], idxs):
        patch.set_facecolor((*colors[weights[i]][:3], 0.4))
    ax_rt.set_xticks([i + 1 for i in idxs])
    ax_rt.set_xticklabels([f"w{weights[i]}\n(n={len(rt_box[i])})" for i in idxs], fontsize=8)
ax_rt.set_yscale("log")
ax_rt.set_xlabel("Pauli weight")
ax_rt.set_ylabel("Per-term runtime (s)")
ax_rt.set_title("Per-term wall-clock runtime distribution")

# ── Save ──────────────────────────────────────────────────────────────────────

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

# ── Text summary ──────────────────────────────────────────────────────────────

print(f"\n{'Weight':>8}  {'JSONL files':>12}  {'NPZ tasks':>10}  "
      f"{'EV sum':>12}  {'Median RT (s)':>14}  {'Median trunc L1':>16}")
print("-" * 80)
for w in weights:
    n_jsonl = len(jsonl_data.get(w, []))
    ev_arr  = np.concatenate(npz_data[w]["values"]) if npz_data[w]["values"] else np.array([])
    rt_arr  = np.concatenate(npz_data[w]["runtimes"]) if npz_data[w]["runtimes"] else np.array([])
    l1_vals = [f["total_l1"] for f in jsonl_data.get(w, []) if not np.isnan(f.get("total_l1", np.nan))]
    n_npz   = len(npz_data[w]["values"])
    ev_sum  = f"{ev_arr.sum():.6f}" if ev_arr.size else "—"
    rt_med  = f"{np.median(rt_arr):.1f}" if rt_arr.size else "—"
    l1_med  = f"{np.median(l1_vals):.3e}" if l1_vals else "—"
    print(f"{w:>8}  {n_jsonl:>12}  {n_npz:>10}  {ev_sum:>12}  {rt_med:>14}  {l1_med:>16}")
