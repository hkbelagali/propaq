# Scale benchmark

    <venv>/bin/python benchmarks/scale/bench_compare.py BASELINE_VENV CANDIDATE_VENV [--out PREFIX]

Runs three workloads (Pauli Ising Trotter, Pauli random circuit, Majorana
fermionic circuit) at about 1e6 and 1e7 peak terms, with 1 and 4 threads,
3 repeats. Every run is a separate process. Output: `PREFIX.md` and `PREFIX.json`
with median and min wall time, peak terms, peak RSS and candidate/baseline ratios.
A run above `--mem-limit-gb` (default 4) is killed and reported.

`--calibrate-venv VENV` prints the peak terms of the configured cutoffs.
For a quick check use `--sizes 1e6 --repeats 1`.
Sizes of 1e8 terms need more than 4 GB and are not part of the default set.
