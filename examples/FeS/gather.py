"""
gather.py — combine per-task results from a parallel FeS_LUCJ array job.

    python gather.py --weight N
"""

import argparse
import glob

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=int, default=1)
args = parser.parse_args()
w = args.weight

pattern = f"results/FeS_LUCJ_w{w}_*of*.npz"
files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No result files matching {pattern}")

all_values, all_n_terms = [], []
ccsd_energy = n_qubits = n_wN = None

for path in files:
    d = np.load(path)
    all_values.append(d["values"])
    all_n_terms.append(d["n_terms"])
    ccsd_energy = float(d["ccsd_energy"])
    n_qubits    = int(d["n_qubits"])
    n_wN        = int(d["n_wN_pauli_terms"])

values  = np.concatenate(all_values)
n_terms = np.concatenate(all_n_terms)
expectation_value = values.sum()

print(f"Tasks gathered:                {len(files)}")
print(f"Total monomials:               {len(values)}")
print(f"Expectation value (weight-{w}): {expectation_value:.10e}")
print(f"CCSD energy:                   {ccsd_energy:.10e}")

out = f"results/FeS_LUCJ_w{w}_gathered.npz"
np.savez(
    out,
    values=values,
    n_terms=n_terms,
    ccsd_energy=ccsd_energy,
    n_qubits=n_qubits,
    n_wN_pauli_terms=n_wN,
)
print(f"Saved {out}")
