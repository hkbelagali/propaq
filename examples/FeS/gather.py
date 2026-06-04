"""
gather.py — reconstruct per-weight expectation value contributions from .out logs.

    python gather.py [--logs-dir logs]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--logs-dir", default="logs")
parser.add_argument("--hamiltonian-cache", default="hamiltonian_cache.npz")
args = parser.parse_args()

LOGS_DIR = Path(args.logs_dir)

FILE_RE    = re.compile(r"FeS-LUCJ-w(\d+)_\d+_(\d+)\.out$")
TASK_RE    = re.compile(r"Task\s+(\d+)/(\d+):")
PARTIAL_RE = re.compile(r"Partial expectation value:\s+([-\d.e+]+)")
CCSD_RE    = re.compile(r"CCSD energy:\s+([-\d.e+]+)")

# Read weight-0 (identity) contribution directly from the Hamiltonian cache
ecore = None
cache_path = Path(args.hamiltonian_cache)
if cache_path.exists():
    cache  = np.load(cache_path, allow_pickle=False)
    paulis = cache["paulis"].astype(str)
    coeffs = cache["coeffs"]
    ecore  = float(coeffs[np.array([all(c == "I" for c in p) for p in paulis])].sum().real)
else:
    print(f"WARNING: hamiltonian cache not found at {cache_path} — weight-0 (ECORE) not included")

weight_files = defaultdict(list)
for path in LOGS_DIR.glob("FeS-LUCJ-w*_*.out"):
    m = FILE_RE.match(path.name)
    if m:
        weight_files[int(m.group(1))].append((int(m.group(2)), path))

if not weight_files:
    raise FileNotFoundError(f"No matching .out files in {LOGS_DIR}")

ccsd_energy = None
weight_results = {}   # weight -> (ev_sum, n_tasks, n_present, missing_ids)

for weight in sorted(weight_files):
    entries = sorted(weight_files[weight])   # sorted by task_id from filename

    ev_sum     = 0.0
    n_tasks    = None
    seen       = set()   # files that exist (even if crashed)
    present    = set()   # files with a valid result line
    incomplete = []

    for file_task_id, path in entries:
        text = path.read_text()
        seen.add(file_task_id)

        task_m    = TASK_RE.search(text)
        partial_m = PARTIAL_RE.search(text)
        ccsd_m    = CCSD_RE.search(text)

        if ccsd_m and ccsd_energy is None:
            ccsd_energy = float(ccsd_m.group(1))

        if task_m:
            n_tasks = int(task_m.group(2))

        if partial_m is None:
            incomplete.append(file_task_id)
            continue

        present.add(file_task_id)
        ev_sum += float(partial_m.group(1))

    if n_tasks is None:
        print(f"WARNING: weight {weight} — could not determine total task count; skipping")
        continue

    not_run = sorted(set(range(n_tasks)) - seen)   # no file at all
    to_rerun = sorted(incomplete) + not_run         # needs resubmission

    if incomplete:
        print(f"WARNING: weight {weight} — {len(incomplete)} log(s) have no result line "
              f"(crashed?): tasks {incomplete[:10]}{'...' if len(incomplete) > 10 else ''}")

    if not_run:
        n_shown = min(len(not_run), 10)
        suffix  = "..." if len(not_run) > 10 else ""
        print(f"WARNING: weight {weight} — {len(not_run)} task(s) never ran: "
              f"{not_run[:n_shown]}{suffix}")

    if to_rerun:
        print(f"         Resubmit with: sbatch --array={','.join(map(str, to_rerun))} "
              f"--export=ALL,WEIGHT={weight} run.sh")

    missing = sorted(set(range(n_tasks)) - present)

    weight_results[weight] = (ev_sum, n_tasks, len(present), missing)

print()
if ccsd_energy is not None:
    print(f"CCSD energy: {ccsd_energy:.10e}")
print()

header = f"{'Weight':>7}  {'Tasks':>7}  {'Present':>7}  {'Missing':>7}  "
header += f"{'EV contribution':>18}  {'Cumulative EV':>18}"
print(header)
print("-" * len(header))

cumulative = 0.0
if ecore is not None:
    cumulative += ecore
    print(f"{'0':>7}  {'—':>7}  {'—':>7}  {'—':>7}  {ecore:>18.10e}  {cumulative:>18.10e}  (ECORE)")
elif 0 not in weight_results:
    print("WARNING: no Hamiltonian build log found — ECORE (weight-0) not included")

for weight in sorted(weight_results):
    ev_sum, n_tasks, n_present, missing = weight_results[weight]
    cumulative += ev_sum
    flag = " *" if missing else ""
    print(f"{weight:>7}  {n_tasks:>7}  {n_present:>7}  {len(missing):>7}  "
          f"{ev_sum:>18.10e}  {cumulative:>18.10e}{flag}")

print()
print(f"Cumulative expectation value: {cumulative:.10e}")
if ccsd_energy is not None:
    print(f"CCSD energy:                  {ccsd_energy:.10e}")
    print(f"Difference:                   {cumulative - ccsd_energy:.10e}")
