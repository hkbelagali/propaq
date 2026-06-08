"""
gather.py — Aggregate per-task npz files into one per-weight file in refined_data/.

    python gather.py [--results-dir results] [--refined-dir refined_data]
                     [--hamiltonian-cache ../hamiltonian_cache.npz]

Reads:  results/FeS_LUCJ_w{N}_{K:05d}of{M:05d}.npz  (per-task, from run_LUCJ.py)
Writes: refined_data/FeS_LUCJ_w{N}.npz               (per-weight, aggregated)
Prints: summary table from the aggregated data.

Fields in each refined_data file:
    ev_sum            — total expectation value contribution for this weight
    values            — all per-monomial EVs concatenated (present tasks, by task_id)
    n_tasks           — total expected task count
    present_task_ids  — sorted task IDs that have results
    missing_task_ids  — sorted task IDs without results
    ccsd_energy       — CCSD reference energy
    n_qubits          — number of qubits
    n_wN_pauli_terms  — number of weight-N Pauli terms
    runtime_seconds   — per-term wall times concatenated across present tasks (NaN if unknown)
    rt_mean, rt_std   — mean and std of per-term runtimes
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir",       default="results")
parser.add_argument("--refined-dir",       default="refined_data")
parser.add_argument("--hamiltonian-cache", default="../hamiltonian_cache.npz")
args = parser.parse_args()

RESULTS_DIR = Path(args.results_dir)
REFINED_DIR = Path(args.refined_dir)

NPZ_RE = re.compile(r"FeS_LUCJ_w(\d+)_(\d{5})of(\d{5})\.npz$")


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


# ── ECORE from Hamiltonian cache ─────────────────────────────────────────────

ecore = None
weight_coeff_mass: dict[int, float] = {}
for cache_name in (args.hamiltonian_cache,):
    p = Path(cache_name)
    if p.exists():
        _c     = np.load(p, allow_pickle=False)
        paulis = _c["paulis"].astype(str)
        coeffs = _c["coeffs"]
        pw = np.array([sum(c != "I" for c in s) for s in paulis])
        ecore = float(coeffs[pw == 0].sum().real)
        for _w in np.unique(pw):
            if _w > 0:
                weight_coeff_mass[int(_w)] = float(np.abs(coeffs[pw == _w]).sum().real)
        break
else:
    print("WARNING: Hamiltonian cache not found — ECORE (weight-0) not included")

# ── Load per-task npz files ───────────────────────────────────────────────────

# weight -> task_id -> {values, runtime, n_tasks, n_wN_pauli_terms, n_qubits, ccsd_energy}
raw: dict[int, dict[int, dict]] = defaultdict(dict)
ccsd_energy = None

for path in RESULTS_DIR.glob("FeS_LUCJ_w*.npz"):
    m = NPZ_RE.match(path.name)
    if not m:
        continue
    w, tid, ntasks = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        d = np.load(path, allow_pickle=False)
        if ccsd_energy is None and "ccsd_energy" in d:
            ccsd_energy = float(d["ccsd_energy"])
        raw[w][tid] = {
            "values":           np.asarray(d["values"], dtype=float),
            "runtime":          np.atleast_1d(np.asarray(d["runtime_seconds"], dtype=float)) if "runtime_seconds" in d else np.array([], dtype=float),
            "n_tasks":          ntasks,
            "n_wN_pauli_terms": int(d["n_wN_pauli_terms"]) if "n_wN_pauli_terms" in d else -1,
            "n_qubits":         int(d["n_qubits"])          if "n_qubits"          in d else -1,
            "ccsd_energy":      float(d["ccsd_energy"])     if "ccsd_energy"       in d else float("nan"),
        }
    except Exception as e:
        print(f"WARNING: could not read {path.name}: {e}")

if not raw:
    raise FileNotFoundError(
        f"No npz result files found in {RESULTS_DIR}. "
        "Run run_LUCJ.py jobs or backfill_npz.py first."
    )

# ── Aggregate and save refined_data/ ─────────────────────────────────────────

REFINED_DIR.mkdir(exist_ok=True)

# weight -> (ev_sum, n_tasks, present_ids, missing_ids, rt_mean, rt_std)
weight_results: dict[int, tuple] = {}

for w in sorted(raw):
    task_map = raw[w]

    n_tasks = next(iter(task_map.values()))["n_tasks"]

    present_ids = sorted(tid for tid in range(n_tasks) if tid in task_map)
    missing_ids = sorted(set(range(n_tasks)) - set(present_ids))

    if missing_ids:
        print(f"WARNING: weight {w} — {len(missing_ids)} task(s) missing from results/")
        print(f"         Resubmit: sbatch --array={_compress_ids(missing_ids)} "
              f"--job-name=FeS-LUCJ-w{w} "
              f"--export=ALL,WEIGHT={w},N_TASKS={n_tasks} run.sh")

    values_all = np.concatenate([task_map[tid]["values"] for tid in present_ids]) \
        if present_ids else np.array([], dtype=float)
    ev_sum = float(values_all.sum())

    runtimes = np.concatenate([task_map[tid]["runtime"] for tid in present_ids]) \
        if present_ids else np.array([], dtype=float)
    valid_rt = runtimes[~np.isnan(runtimes)]
    rt_mean  = float(np.mean(valid_rt)) if valid_rt.size else float("nan")
    rt_std   = float(np.std(valid_rt))  if valid_rt.size else float("nan")

    # Metadata from the first present task
    first = task_map[present_ids[0]] if present_ids else {}
    n_wN  = first.get("n_wN_pauli_terms", -1)
    nq    = first.get("n_qubits", -1)
    ce    = first.get("ccsd_energy", float("nan"))
    if ccsd_energy is None and not np.isnan(ce):
        ccsd_energy = ce

    out_path = REFINED_DIR / f"FeS_LUCJ_w{w}.npz"
    np.savez(
        out_path,
        ev_sum           = np.float64(ev_sum),
        values           = values_all,
        n_tasks          = np.int64(n_tasks),
        present_task_ids = np.array(present_ids, dtype=np.int64),
        missing_task_ids = np.array(missing_ids, dtype=np.int64),
        ccsd_energy      = np.float64(ce),
        n_qubits         = np.int64(nq),
        n_wN_pauli_terms = np.int64(n_wN),
        runtime_seconds  = runtimes,
        rt_mean          = np.float64(rt_mean),
        rt_std           = np.float64(rt_std),
    )

    present_terms = len(values_all)
    if not missing_ids:
        missing_terms_str = "0"
    elif present_ids:
        missing_terms_str = f"~{round(present_terms / len(present_ids) * len(missing_ids))}"
    else:
        missing_terms_str = "?"

    weight_results[w] = (ev_sum, n_tasks, present_ids, missing_ids, rt_mean, rt_std,
                         present_terms, missing_terms_str)

print(f"Saved {len(weight_results)} weight file(s) to {REFINED_DIR}/")

# ── Print summary ─────────────────────────────────────────────────────────────

print()
if ccsd_energy is not None:
    print(f"CCSD energy: {ccsd_energy:.6f}")
print()

RTW = 10
TW  = 8
CMW = 16
EVW = 16
header = (
    f"{'Weight':>7}  {'Pres.T':>{TW}}  {'Miss.T':>{TW}}  "
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
        f"{'0':>7}  {blank:>{TW}}  {blank:>{TW}}  "
        f"{blank:>{CMW}}  {ecore:>{EVW}.6f}  {cumulative:>{EVW}.6f}  "
        f"{blank:>{RTW}}  {blank:>{RTW}}  (ECORE)"
    )
elif 0 not in weight_results:
    print("WARNING: no Hamiltonian cache found — ECORE not included")

for w in sorted(weight_results):
    ev_sum, n_tasks, present_ids, missing_ids, rt_mean, rt_std, present_terms, missing_terms_str = weight_results[w]
    cumulative += ev_sum
    flag = " *" if missing_ids else ""
    cm = weight_coeff_mass.get(w)
    cm_str = f"{cm:>{CMW}.6f}" if cm is not None else f"{blank:>{CMW}}"
    print(
        f"{w:>7}  {present_terms:>{TW}}  {missing_terms_str:>{TW}}  "
        f"{cm_str}  {ev_sum:>{EVW}.6f}  {cumulative:>{EVW}.6f}  "
        f"{_fmt_time(rt_mean):>{RTW}}  {_fmt_time(rt_std):>{RTW}}{flag}"
    )

print()
print(f"Cumulative expectation value: {cumulative:.6f}")
if ccsd_energy is not None:
    print(f"CCSD energy:                  {ccsd_energy:.6f}")
    print(f"Difference:                   {cumulative - ccsd_energy:.6f}")
