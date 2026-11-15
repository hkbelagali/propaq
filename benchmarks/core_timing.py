"""Single-thread timing of the golden cases.

Run it with the installed wheel of one build, from any directory:
    <venv>/bin/python -P benchmarks/core_timing.py out.json [--repeat N] [--heavy]

It loads tests/golden/generate_golden.py by path, so the source tree does
not shadow the installed package. Each case runs N times. The report keeps
the best time. Compare two builds with --compare a.json b.json.
"""

import importlib.util
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
HEAVY = {"pauli_ising4x4_s5_noise_w8"}


def load_cases():
    path = HERE.parent / "tests" / "golden" / "generate_golden.py"
    spec = importlib.util.spec_from_file_location("generate_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(out_path, repeat, heavy):
    gg = load_cases()
    result = {}
    for name, kind, qc, op, noise, tr in gg.cases():
        if name in HEAVY and not heavy:
            continue
        times = []
        n_terms = None
        for _ in range(repeat if name not in HEAVY else 1):
            t0 = time.perf_counter()
            res = gg.run_case(kind, qc, op, noise, tr)
            times.append(time.perf_counter() - t0)
            n_terms = res["n_terms"][-1] if res["n_terms"] else None
        result[name] = {"best_s": min(times), "all_s": times, "final_terms": n_terms}
        print(f"{name}: best {min(times):.3f} s, final terms {n_terms}", flush=True)
    pathlib.Path(out_path).write_text(json.dumps(result, indent=1))


def compare(a_path, b_path):
    a = json.loads(pathlib.Path(a_path).read_text())
    b = json.loads(pathlib.Path(b_path).read_text())
    worst = 0.0
    total_a = total_b = 0.0
    for name in sorted(a):
        if name not in b:
            continue
        ratio = b[name]["best_s"] / a[name]["best_s"]
        worst = max(worst, ratio)
        total_a += a[name]["best_s"]
        total_b += b[name]["best_s"]
        print(f"{name:55s} {a[name]['best_s']:8.3f} {b[name]['best_s']:8.3f}  x{ratio:.3f}")
    print(f"total: {total_a:.3f} s vs {total_b:.3f} s, x{total_b / total_a:.3f}; worst case x{worst:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        compare(args[1], args[2])
    else:
        repeat = int(args[args.index("--repeat") + 1]) if "--repeat" in args else 3
        measure(args[0], repeat, "--heavy" in args)
