"""
Aggregate per-task npz files into one per-order file in refined_data/.

    ev_sum            — total expectation value contribution for this order bucket
    values            — all per-monomial EVs concatenated (present tasks, by task_id)
    n_tasks           — total expected task count
    present_task_ids  — sorted task IDs that have results
    missing_task_ids  — sorted task IDs without results
    n_qubits          — number of qubits
    n_oN_pauli_terms  — number of order-N Pauli terms
    runtime_seconds   — per-term wall times concatenated across present tasks
    rt_mean, rt_std   — mean and std of per-term runtimes
"""

import argparse
import re
from collections import defaultdict
from math import floor, log10
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--natoms",       type=int, required=True,  help="Number of hydrogen atoms")
parser.add_argument("--connectivity", type=str, default="heavy-hex",
                    choices=["square", "heavy-hex", "all-to-all"],
                    help="Connectivity topology used in build/run (default: heavy-hex)")
parser.add_argument("--results-dir",  default="results",      help="Directory containing per-task npz files")
parser.add_argument("--refined-dir",  default="refined_data", help="Output directory for aggregated files")
args = parser.parse_args()

natoms       = args.natoms
connectivity = args.connectivity
RESULTS_DIR  = Path(args.results_dir)
REFINED_DIR  = Path(args.refined_dir)

NPZ_RE = re.compile(
    rf"Hchain_n{natoms}_{re.escape(connectivity)}_o(-?\d+)_(\d{{5}})of(\d{{5}})\.npz$"
)

def _compress_ids(ids: list[int]) -> str:
    """Convert a sorted list of ints to SLURM range notation, e.g. [0,1,2,5] -> '0-2,5'."""
    if not ids:
        return ""
    parts = []
    start = end = ids[0]
    for n in ids[1:]:
        if n == end + 1:
            end = n
        else:
            parts.append(f"{start}-{end}" if end > start else str(start))
            start = end = n
    parts.append(f"{start}-{end}" if end > start else str(start))
    return ",".join(parts)

def _fmt_time(sec: float) -> str:
    if np.isnan(sec):
        return "—"
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

ecore: float | None = None
e_ccsd: float | None = None
order_coeff_mass: dict[int, float] = {}

cache_path = Path(f"n{natoms}/{connectivity}/hamiltonian_cache.npz")
if cache_path.exists():
    _c      = np.load(cache_path, allow_pickle=False)
    paulis  = _c["paulis"].astype(str)
    coeffs  = _c["coeffs"].real
    pw      = np.array([sum(ch != "I" for ch in s) for s in paulis])
    ecore   = float(coeffs[pw == 0].sum())
    if "e_ccsd" in _c:
        e_ccsd = float(_c["e_ccsd"])
    for c in coeffs[pw > 0]:
        if abs(c) > 0:
            o = floor(log10(abs(c)))
            order_coeff_mass[o] = order_coeff_mass.get(o, 0.0) + abs(c)
else:
    print(f"WARNING: Hamiltonian cache not found at {cache_path} — ECORE not included")

raw: dict[int, dict[int, dict]] = defaultdict(dict)

for path in RESULTS_DIR.glob(f"Hchain_n{natoms}_{connectivity}_o*.npz"):
    m = NPZ_RE.match(path.name)
    if not m:
        continue
    order, tid, ntasks = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        d = np.load(path, allow_pickle=False)
        raw[order][tid] = {
            "values": np.asarray(d["values"], dtype=float),
            "runtime": np.atleast_1d(np.asarray(d["runtime_seconds"], dtype=float))
                                if "runtime_seconds" in d else np.array([], dtype=float),
            "n_tasks": ntasks,
            "n_oN_pauli_terms": int(d["n_oN_pauli_terms"]) if "n_oN_pauli_terms" in d else -1,
            "n_qubits": int(d["n_qubits"])          if "n_qubits"          in d else -1,
            "damping": float(d["damping"])          if "damping"          in d else None,
            "coeff_cutoff": float(d["coeff_cutoff"])     if "coeff_cutoff"     in d else None,
        }
    except Exception as e:
        print(f"WARNING: could not read {path.name}: {e}")

if not raw:
    raise FileNotFoundError(
        f"No matching npz files found in {RESULTS_DIR} for "
        f"H{natoms} ({connectivity}). Run run_LUCJ.py jobs first."
    )

REFINED_DIR.mkdir(exist_ok=True)

order_results: dict[int, tuple] = {}

for order in sorted(raw, reverse=True):
    task_map = raw[order]

    n_tasks     = next(iter(task_map.values()))["n_tasks"]
    present_ids = sorted(tid for tid in range(n_tasks) if tid in task_map)
    missing_ids = sorted(set(range(n_tasks)) - set(present_ids))

    if missing_ids:
        print(f"WARNING: order {order} — {len(missing_ids)} task(s) missing")
        print(f"         Resubmit: sbatch --array={_compress_ids(missing_ids)} "
              f"--job-name=Hchain-n{natoms}-o{order} "
              f"--export=ALL,NATOMS={natoms},CONNECTIVITY={connectivity},"
              f"ORDER={order},N_TASKS={n_tasks} run.sh")

    values_all = np.concatenate([task_map[tid]["values"] for tid in present_ids]) \
        if present_ids else np.array([], dtype=float)
    ev_sum = float(values_all.sum())

    runtimes  = np.concatenate([task_map[tid]["runtime"] for tid in present_ids]) \
        if present_ids else np.array([], dtype=float)
    valid_rt  = runtimes[~np.isnan(runtimes)]
    rt_mean   = float(np.mean(valid_rt)) if valid_rt.size else float("nan")
    rt_std    = float(np.std(valid_rt))  if valid_rt.size else float("nan")

    first        = task_map[present_ids[0]] if present_ids else {}
    n_oN         = first.get("n_oN_pauli_terms", -1)
    nq           = first.get("n_qubits", -1)
    damping      = first.get("damping")
    coeff_cutoff = first.get("coeff_cutoff")

    out_path = REFINED_DIR / f"Hchain_n{natoms}_{connectivity}_o{order}.npz"
    np.savez(
        out_path,
        ev_sum = np.float64(ev_sum),
        values = values_all,
        n_tasks = np.int64(n_tasks),
        present_task_ids = np.array(present_ids, dtype=np.int64),
        missing_task_ids = np.array(missing_ids, dtype=np.int64),
        n_qubits = np.int64(nq),
        n_oN_pauli_terms = np.int64(n_oN),
        runtime_seconds = runtimes,
        rt_mean = np.float64(rt_mean),
        rt_std = np.float64(rt_std),
        **({} if damping      is None else {"damping":      np.float64(damping)}),
        **({} if coeff_cutoff is None else {"coeff_cutoff": np.float64(coeff_cutoff)}),
    )

    present_terms = len(values_all)
    if not missing_ids:
        missing_terms_str = "0"
    elif present_ids:
        missing_terms_str = f"~{round(present_terms / len(present_ids) * len(missing_ids))}"
    else:
        missing_terms_str = "?"

    order_results[order] = (ev_sum, n_tasks, present_ids, missing_ids, rt_mean, rt_std,
                             present_terms, missing_terms_str, damping, coeff_cutoff)

print(f"Saved {len(order_results)} order file(s) to {REFINED_DIR}/")

print()
print(f"H{natoms} ({connectivity})")
print()

RTW = 10
TW  = 8
CMW = 16
EVW = 16
header = (
    f"{'Order':>7}  {'Pres.T':>{TW}}  {'Miss.T':>{TW}}  "
    f"{'|c| mass':>{CMW}}  {'EV contribution':>{EVW}}  {'Cumulative EV':>{EVW}}  "
    f"{'RT mean':>{RTW}}  {'RT std':>{RTW}}"
)
print(header)
print("-" * len(header))

cumulative = 0.0
blank = "—"
if ecore is not None:
    cumulative += ecore
    print(
        f"{'ECORE':>7}  {blank:>{TW}}  {blank:>{TW}}  "
        f"{blank:>{CMW}}  {ecore:>{EVW}.6f}  {cumulative:>{EVW}.6f}  "
        f"{blank:>{RTW}}  {blank:>{RTW}}"
    )
else:
    print("WARNING: no Hamiltonian cache found — ECORE not included")

for order in sorted(order_results, reverse=True):
    ev_sum, n_tasks, present_ids, missing_ids, rt_mean, rt_std, present_terms, missing_terms_str, damping, coeff_cutoff = order_results[order]
    cumulative += ev_sum
    flag   = " *" if missing_ids else ""
    cm     = order_coeff_mass.get(order)
    cm_str = f"{cm:>{CMW}.6f}" if cm is not None else f"{blank:>{CMW}}"
    params = ""
    if damping is not None or coeff_cutoff is not None:
        parts = []
        if damping      is not None: parts.append(f"damping={damping}")
        if coeff_cutoff is not None: parts.append(f"cutoff={coeff_cutoff:.0e}")
        params = f"  [{', '.join(parts)}]"
    print(
        f"{order:>7}  {present_terms:>{TW}}  {missing_terms_str:>{TW}}  "
        f"{cm_str}  {ev_sum:>{EVW}.6f}  {cumulative:>{EVW}.6f}  "
        f"{_fmt_time(rt_mean):>{RTW}}  {_fmt_time(rt_std):>{RTW}}{flag}{params}"
    )

print()
print(f"Cumulative expectation value: {cumulative:.6f}")
if e_ccsd is not None:
    print(f"CCSD energy:                  {e_ccsd:.6f}")
