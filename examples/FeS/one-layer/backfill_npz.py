"""
backfill_npz.py — One-time utility: backfill npz result files from existing log data.

For each completed task found in logs/:
  - If the corresponding npz does not exist: create it from the log data.
  - If the npz exists but is missing 'runtime_seconds': patch it from the err file.

Run once to populate results/ from logs/ for any tasks that completed but whose
npz is missing or lacks runtime_seconds.

    python backfill_npz.py [--logs-dir logs] [--results-dir results] [--dry-run]
"""

import argparse
import re
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--logs-dir",    default="logs")
parser.add_argument("--results-dir", default="results")
parser.add_argument("--dry-run",     action="store_true",
                    help="Print what would be done without writing anything")
args = parser.parse_args()

LOGS_DIR    = Path(args.logs_dir)
RESULTS_DIR = Path(args.results_dir)
DRY_RUN     = args.dry_run

# Filename patterns
OUT_RE = re.compile(r"FeS-LUCJ-w(\d+)_(\d+)\.out$")           # w{weight}_{task_id}
NPZ_RE = re.compile(r"FeS_LUCJ_w(\d+)_(\d{5})of(\d{5})\.npz$")

# Content patterns
EV_RE       = re.compile(r"^Expectation value:\s+([-\d.e+]+)", re.MULTILINE)
PARTIAL_RE  = re.compile(r"^Partial expectation value:\s+([-\d.e+]+)", re.MULTILINE)
CCSD_RE     = re.compile(r"^CCSD energy:\s+([-\d.e+]+)", re.MULTILINE)
TASK_RE     = re.compile(r"^Task\s+(\d+)/(\d+):", re.MULTILINE)
WEIGHT_RE   = re.compile(r"^Weight-(\d+) terms:\s+(\d+)", re.MULTILINE)
NQUBITS_RE  = re.compile(r"^Number of qubits:\s+(\d+)", re.MULTILINE)

# tqdm outer progress bar: "weight-N task K: XX%|...| n/N [elapsed<..."
TQDM_RE     = re.compile(
    r"weight-(\d+) task (\d+):\s+\d+%\|[^|]*\|\s+(\d+)/\d+\s+\[(\d+:\d{2}(?::\d{2})?)<"
)


def _parse_elapsed(s: str) -> float:
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _parse_out(path: Path) -> dict | None:
    """Parse a .out log file. Returns None if task was incomplete."""
    text = path.read_text()
    if not PARTIAL_RE.search(text):
        return None  # task did not complete

    r: dict = {}

    wm = WEIGHT_RE.search(text)
    if not wm:
        return None
    r["weight"]           = int(wm.group(1))
    r["n_wN_pauli_terms"] = int(wm.group(2))

    tm = TASK_RE.search(text)
    if not tm:
        return None
    r["task_id"] = int(tm.group(1))
    r["n_tasks"]  = int(tm.group(2))

    cm = CCSD_RE.search(text)
    r["ccsd_energy"] = float(cm.group(1)) if cm else float("nan")

    nm = NQUBITS_RE.search(text)
    r["n_qubits"] = int(nm.group(1)) if nm else -1

    r["values"] = np.array([float(x) for x in EV_RE.findall(text)])

    return r


def _parse_err_runtimes(path: Path) -> np.ndarray | None:
    """Return per-term runtimes (seconds) derived from consecutive tqdm elapsed diffs."""
    try:
        text = path.read_text()
    except OSError:
        return None
    steps: dict[int, float] = {}
    for line in re.split(r"[\r\n]", text):
        mx = TQDM_RE.search(line)
        if not mx:
            continue
        n_done  = int(mx.group(3))
        elapsed = _parse_elapsed(mx.group(4))
        steps[n_done] = elapsed  # latest value for each step wins
    if not steps:
        return None
    baseline = steps.get(0, 0.0)
    n_max    = max(steps)
    runtimes = []
    prev     = baseline
    for n in range(1, n_max + 1):
        if n in steps:
            runtimes.append(steps[n] - prev)
            prev = steps[n]
        else:
            runtimes.append(float("nan"))
    return np.array(runtimes) if runtimes else None


def _npz_path(results_dir: Path, weight: int, task_id: int, n_tasks: int) -> Path:
    return results_dir / f"FeS_LUCJ_w{weight}_{task_id:05d}of{n_tasks:05d}.npz"


def _err_sibling(out_path: Path) -> Path:
    return out_path.with_suffix(".err")


# ── Build index of existing npz files ────────────────────────────────────────

existing_npz: set[Path] = set()
npz_has_runtime: set[Path] = set()

for p in RESULTS_DIR.glob("FeS_LUCJ_w*.npz"):
    existing_npz.add(p)
    try:
        d = np.load(p, allow_pickle=False)
        if "runtime_seconds" in d:
            rt = np.atleast_1d(d["runtime_seconds"].astype(float))
            if not np.all(np.isnan(rt)):
                npz_has_runtime.add(p)
    except Exception:
        pass

print(f"Found {len(existing_npz)} existing npz files "
      f"({len(npz_has_runtime)} already have runtime_seconds)")

# ── Process log files ─────────────────────────────────────────────────────────

# (weight, task_id, n_tasks) -> {"out_path": Path, "data": dict}
candidates: dict[tuple, dict] = {}

for path in LOGS_DIR.glob("FeS-LUCJ-w*.out"):
    m = OUT_RE.match(path.name)
    if not m:
        continue
    w, tid = int(m.group(1)), int(m.group(2))
    r = _parse_out(path)
    if r is None:
        continue
    key = (w, tid, r["n_tasks"])
    candidates[key] = {"out_path": path, "data": r}

print(f"Found {len(candidates)} completed tasks in logs")

# ── Create or patch npz files ─────────────────────────────────────────────────

n_created = 0
n_patched = 0
n_skipped = 0

RESULTS_DIR.mkdir(exist_ok=True)

for (weight, task_id, n_tasks), info in sorted(candidates.items()):
    r        = info["data"]
    out_path = info["out_path"]
    npz_path = _npz_path(RESULTS_DIR, weight, task_id, n_tasks)
    err_path = _err_sibling(out_path)

    runtimes = _parse_err_runtimes(err_path)
    n_vals   = len(r["values"])
    rt_arr   = runtimes if runtimes is not None else np.full(n_vals, float("nan"))

    if npz_path not in existing_npz:
        # Create new npz from log data
        if DRY_RUN:
            n_rt = 0 if runtimes is None else int(np.sum(~np.isnan(runtimes)))
            print(f"[dry-run] would create {npz_path.name}"
                  + (f"  {n_rt}/{n_vals} term runtimes" if runtimes is not None else ""))
        else:
            np.savez(
                npz_path,
                values=r["values"],
                n_terms=np.full(n_vals, -1, dtype=np.int64),  # not logged
                ccsd_energy=r["ccsd_energy"],
                n_qubits=r["n_qubits"],
                n_wN_pauli_terms=r["n_wN_pauli_terms"],
                task_id=task_id,
                n_tasks=n_tasks,
                runtime_seconds=rt_arr,
            )
            n_rt = int(np.sum(~np.isnan(rt_arr)))
            print(f"created  {npz_path.name}  {n_rt}/{n_vals} term runtimes")
        n_created += 1

    elif npz_path not in npz_has_runtime and runtimes is not None:
        # Patch existing npz with per-term runtime_seconds
        if DRY_RUN:
            n_rt = int(np.sum(~np.isnan(runtimes)))
            print(f"[dry-run] would patch  {npz_path.name}  {n_rt} term runtimes")
        else:
            d = dict(np.load(npz_path, allow_pickle=False))
            d["runtime_seconds"] = runtimes
            np.savez(npz_path, **d)
            npz_has_runtime.add(npz_path)
            n_rt = int(np.sum(~np.isnan(runtimes)))
            print(f"patched  {npz_path.name}  {n_rt} term runtimes")
        n_patched += 1

    else:
        n_skipped += 1

print()
action = "[dry-run] would " if DRY_RUN else ""
print(f"{action}create:  {n_created}")
print(f"{action}patch:   {n_patched}")
print(f"skipped: {n_skipped}  (already complete or no err file)")
