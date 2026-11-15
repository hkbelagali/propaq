"""Compare two propaq builds on fixed workloads at about 1e6 and 1e7 peak terms.

Usage:
    python bench_compare.py BASELINE_VENV CANDIDATE_VENV [options]

Each venv holds one propaq build. Every measurement runs in its own process
with ``<venv>/bin/python -P`` (so the source tree never shadows the wheel).
The worker times only ``expectation_value``. The parent reads the peak RSS
of the child from /proc and kills it when it passes ``--mem-limit-gb``.

For each workload, size, thread count and venv, the report gives the median
wall time of the repeats, the peak term count, the peak RSS, and the ratio
candidate / baseline of the median time and of the peak RSS. Baseline and candidate runs alternate, so slow load drift on a shared machine
hits both. The ratio is the main result. A ratio below 1 means the candidate is faster.

Results are written as markdown and JSON. Passing the same venv twice is a
noise check: the ratios must be close to 1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Workloads. Each size entry sets the cutoff that gives about the target
# peak term count (measured on v0.1.8: 0.9M-1.2M and 10.3M-10.5M).
# `--calibrate-venv VENV` prints the peak terms for these cutoffs.
# ---------------------------------------------------------------------------
WORKLOADS = {
    "pauli_ising": {
        "params": {"nx": 4, "ny": 4, "steps": 6},
        "sizes": {"1e6": {"coeff_cutoff": 2e-7}, "1e7": {"coeff_cutoff": 3e-9}},
    },
    "pauli_random": {
        "params": {"n": 16, "layers": 8, "seed": 3},
        "sizes": {"1e6": {"coeff_cutoff": 2e-4}, "1e7": {"coeff_cutoff": 3e-5}},
    },
    "majorana_fermionic": {
        "params": {"n": 14, "layers": 6, "seed": 21},
        "sizes": {"1e6": {"coeff_cutoff": 3e-5}, "1e7": {"coeff_cutoff": 2e-6}},
    },
}


# ---------------------------------------------------------------------------
# Worker side: runs inside the venv under test.
# ---------------------------------------------------------------------------
def _label(n, ops):
    s = ["I"] * n
    for q, ch in ops.items():
        s[n - 1 - q] = ch
    return "".join(s)


def build_case(name, p):
    """Return (kind, qiskit circuit, SparsePauliOp)."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZZGate, XXPlusYYGate
    from qiskit.quantum_info import SparsePauliOp

    if name == "pauli_ising":
        nx, ny, steps, dt, h = p["nx"], p["ny"], p["steps"], 0.1, 1.0
        n = nx * ny
        qc = QuantumCircuit(n)
        edges = []
        for x in range(nx):
            for y in range(ny):
                q = x * ny + y
                if x + 1 < nx:
                    edges.append((q, (x + 1) * ny + y))
                if y + 1 < ny:
                    edges.append((q, x * ny + y + 1))
        for _ in range(steps):
            for a, b in edges:
                qc.append(RZZGate(2 * dt), [a, b])
            for q in range(n):
                qc.rx(2 * h * dt, q)
        op = SparsePauliOp.from_list([(_label(n, {n // 3: "Z"}), 1.0)])
        return "pauli", qc, op
    rng = np.random.default_rng(p["seed"])
    n = p["n"]
    qc = QuantumCircuit(n)
    if name == "pauli_random":
        for _ in range(p["layers"]):
            for q in range(n):
                qc.rx(float(rng.uniform(0, np.pi)), q)
                qc.rz(float(rng.uniform(0, np.pi)), q)
            for q in range(0, n - 1, 2):
                qc.append(RZZGate(float(rng.uniform(0, np.pi))), [q, q + 1])
            for q in range(1, n - 1, 2):
                qc.cx(q, q + 1)
        op = SparsePauliOp.from_list([(_label(n, {0: "Z"}), 1.0), (_label(n, {n // 2: "X"}), 0.5)])
        return "pauli", qc, op
    if name == "majorana_fermionic":
        for _ in range(p["layers"]):
            for q in range(0, n - 1, 2):
                qc.append(XXPlusYYGate(float(rng.uniform(0, np.pi)), float(rng.uniform(0, np.pi))), [q, q + 1])
            for q in range(1, n - 1, 2):
                qc.append(XXPlusYYGate(float(rng.uniform(0, np.pi)), 0.0), [q, q + 1])
            for q in range(n):
                qc.rz(float(rng.uniform(0, np.pi)), q)
            for q in range(0, n - 1, 3):
                qc.cz(q, q + 1)
        op = SparsePauliOp.from_list([(_label(n, {q: "Z"}), 1.0) for q in range(n)] + [(_label(n, {0: "X", 1: "X"}), 0.25)])
        return "majorana", qc, op
    raise ValueError(name)


def worker(spec):
    import resource
    import warnings

    warnings.simplefilter("ignore")
    import propaq
    from propaq.circuits import MajoranaCircuit, PauliCircuit
    from propaq.datatypes import MajoranaTermSum, PauliTermSum
    from propaq.propagators import MajoranaPropagator, PauliPropagator
    from propaq.truncation import TruncationPolicy

    kind, qc, op = build_case(spec["workload"], spec["params"])
    trunc = TruncationPolicy(coeff_cutoff=spec["coeff_cutoff"])
    if kind == "pauli":
        prop = PauliPropagator(truncation=trunc, n_threads=spec["threads"], pin_threads=False)
        obs, circ = PauliTermSum.from_sparse_pauli_op(op), PauliCircuit.from_qiskit(qc)
    else:
        prop = MajoranaPropagator(truncation=trunc, n_threads=spec["threads"], pin_threads=False)
        obs = MajoranaTermSum.from_sparse_pauli_op(op)
        circ = MajoranaCircuit.from_qiskit(qc, n_modes=2 * qc.num_qubits)
    t0 = time.perf_counter()
    res = prop.expectation_value(obs, circ, initial_state=0)
    wall = time.perf_counter() - t0
    out = {
        "version": propaq.__version__,
        "wall_s": wall,
        "peak_terms": int(max(res.n_terms)),
        "final_terms": int(res.n_terms[-1]),
        "expectation_value": float(res.expectation_value),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    print("RESULT " + json.dumps(out))


# ---------------------------------------------------------------------------
# Parent side.
# ---------------------------------------------------------------------------
def _vm_hwm_mb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def run_one(venv, spec, mem_limit_mb, timeout_s):
    cmd = [str(Path(venv) / "bin" / "python"), "-P", str(Path(__file__).resolve()), "--worker", json.dumps(spec)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start, peak, status = time.time(), 0.0, "ok"
    while proc.poll() is None:
        peak = max(peak, _vm_hwm_mb(proc.pid))
        if peak > mem_limit_mb:
            proc.kill()
            status = "killed: memory limit"
        elif time.time() - start > timeout_s:
            proc.kill()
            status = "killed: timeout"
        time.sleep(0.2)
    out, err = proc.communicate()
    result = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
    if result is None and status == "ok":
        status = "failed: " + err.strip().splitlines()[-1] if err.strip() else "failed"
    if result is not None:
        result["maxrss_mb"] = max(result["maxrss_mb"], peak)
    return status, result


def summarize(runs, status):
    if not runs:
        return {"status": status, "runs": []}
    return {
        "status": status,
        "runs": runs,
        "median_wall_s": statistics.median(r["wall_s"] for r in runs),
        "min_wall_s": min(r["wall_s"] for r in runs),
        "peak_terms": runs[0]["peak_terms"],
        "terms_agree": len({r["peak_terms"] for r in runs}) == 1,
        "expectation_value": runs[0]["expectation_value"],
        "peak_rss_mb": max(r["maxrss_mb"] for r in runs),
        "version": runs[0]["version"],
    }


def measure_pair(base_venv, cand_venv, spec, repeats, mem_limit_mb, timeout_s):
    """Run baseline and candidate alternately, so load drift hits both the same way."""
    runs = {"baseline": [], "candidate": []}
    status = {"baseline": "ok", "candidate": "ok"}
    for _ in range(repeats):
        for key, venv in (("baseline", base_venv), ("candidate", cand_venv)):
            if status[key] != "ok":
                continue
            status[key], r = run_one(venv, spec, mem_limit_mb, timeout_s)
            if r is not None:
                runs[key].append(r)
    return summarize(runs["baseline"], status["baseline"]), summarize(runs["candidate"], status["candidate"])


def render(rows, base_venv, cand_venv, meta):
    lines = [
        "# propaq scale benchmark: candidate vs baseline",
        "",
        f"* baseline: `{base_venv}` (propaq {meta['baseline_version']})",
        f"* candidate: `{cand_venv}` (propaq {meta['candidate_version']})",
        f"* repeats: {meta['repeats']}; wall time is the median of the repeats; RSS is the maximum.",
        "* ratio = candidate / baseline. Below 1 means the candidate is faster or smaller.",
        "* baseline and candidate runs alternate. The min ratio resists load from other processes.",
        "",
        "| workload | size | thr | peak terms | base time s | cand time s | time ratio (median) | time ratio (min) | base RSS MB | cand RSS MB | RSS ratio | value diff | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, c = r["baseline"], r["candidate"]
        if "median_wall_s" not in b or "median_wall_s" not in c:
            note = f"baseline {b['status']}; candidate {c['status']}"
            lines.append(f"| {r['workload']} | {r['size']} | {r['threads']} | - | - | - | - | - | - | - | - | - | {note} |")
            continue
        note = []
        if b["peak_terms"] != c["peak_terms"]:
            note.append("PEAK TERMS DIFFER")
        if not (b["terms_agree"] and c["terms_agree"]):
            note.append("terms vary between repeats")
        diff = abs(b["expectation_value"] - c["expectation_value"])
        lines.append(
            f"| {r['workload']} | {r['size']} | {r['threads']} | {b['peak_terms']:,} | {b['median_wall_s']:.2f} | "
            f"{c['median_wall_s']:.2f} | {c['median_wall_s'] / b['median_wall_s']:.3f} | {c['min_wall_s'] / b['min_wall_s']:.3f} | {b['peak_rss_mb']:.0f} | "
            f"{c['peak_rss_mb']:.0f} | {c['peak_rss_mb'] / b['peak_rss_mb']:.3f} | {diff:.1e} | {'; '.join(note)} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline_venv", nargs="?")
    ap.add_argument("candidate_venv", nargs="?")
    ap.add_argument("--worker", help="internal: run one spec (JSON) and print the result")
    ap.add_argument("--sizes", nargs="+", default=["1e6", "1e7"], choices=["1e6", "1e7"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=list(WORKLOADS))
    ap.add_argument("--mem-limit-gb", type=float, default=4.0)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--out", default="bench_compare", help="output prefix; writes <out>.md and <out>.json")
    ap.add_argument("--calibrate-venv", help="print peak terms for the configured cutoffs using this venv, 4 threads")
    a = ap.parse_args()
    if a.worker:
        worker(json.loads(a.worker))
        return
    mem_mb = a.mem_limit_gb * 1024
    if a.calibrate_venv:
        for w in a.workloads:
            for size in a.sizes:
                spec = {"workload": w, "params": WORKLOADS[w]["params"], "threads": 4, **WORKLOADS[w]["sizes"][size]}
                st, r = run_one(a.calibrate_venv, spec, mem_mb, a.timeout_s)
                print(w, size, spec["coeff_cutoff"], st, r and (r["peak_terms"], round(r["wall_s"], 2), round(r["maxrss_mb"])), flush=True)
        return
    if not (a.baseline_venv and a.candidate_venv):
        ap.error("two venv paths are required")
    rows = []
    versions = {}
    for w in a.workloads:
        for size in a.sizes:
            for thr in a.threads:
                spec = {"workload": w, "params": WORKLOADS[w]["params"], "threads": thr, **WORKLOADS[w]["sizes"][size]}
                row = {"workload": w, "size": size, "threads": thr, "spec": spec}
                row["baseline"], row["candidate"] = measure_pair(
                    a.baseline_venv, a.candidate_venv, spec, a.repeats, mem_mb, a.timeout_s
                )
                versions["baseline"] = row["baseline"].get("version", versions.get("baseline", "?"))
                versions["candidate"] = row["candidate"].get("version", versions.get("candidate", "?"))
                rows.append(row)
                b, c = row["baseline"], row["candidate"]
                if "median_wall_s" in b and "median_wall_s" in c:
                    print(f"{w} {size} thr={thr}: base {b['median_wall_s']:.2f}s cand {c['median_wall_s']:.2f}s "
                          f"ratio {c['median_wall_s'] / b['median_wall_s']:.3f} (min {c['min_wall_s'] / b['min_wall_s']:.3f}) terms {b['peak_terms']:,}", flush=True)
                else:
                    print(f"{w} {size} thr={thr}: baseline {b['status']}; candidate {c['status']}", flush=True)
    meta = {"baseline_version": versions.get("baseline", "?"), "candidate_version": versions.get("candidate", "?"), "repeats": a.repeats}
    Path(a.out + ".json").write_text(json.dumps({"meta": meta, "baseline_venv": a.baseline_venv, "candidate_venv": a.candidate_venv, "rows": rows}, indent=1))
    Path(a.out + ".md").write_text(render(rows, a.baseline_venv, a.candidate_venv, meta))
    print("wrote", a.out + ".md", a.out + ".json")


if __name__ == "__main__":
    main()
