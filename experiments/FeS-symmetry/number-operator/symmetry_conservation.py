"""
Compute <psi|U^dag N U|psi> where U implements the LUCJ circuit and N is the number operator. 

The number of particles is trivially preserved by the LUCJ circuit, so this is a sanity check.
"""

import argparse
import os
import time

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp

from propaq.circuits import MajoranaCircuit
from propaq.datatypes import MajoranaTermSum
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators import MajoranaPropagator

parser = argparse.ArgumentParser()
parser.add_argument("--cutoff",    type=float, default=1e-10,
                    help="Coefficient truncation cutoff")
parser.add_argument("--n-threads", type=int,   default=128)
args = parser.parse_args()

with open("../FeS_LUCJ_circuit.qpy", "rb") as f:
    compiled = qpy.load(f)[0]

n_qubits = compiled.num_qubits
n_modes  = 2 * n_qubits
mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=n_modes)

cache = np.load("../compiled_hamiltonian_cache.npz", allow_pickle=False)
ccsd_energy = float(cache["ccsd_energy"])

print(f"n_qubits    = {n_qubits}")
print(f"n_modes     = {n_modes}")
print(f"CCSD energy = {ccsd_energy:.10e}")

n_op_terms = [("I" * n_qubits, n_qubits * 0.5)]
for j in range(n_qubits):
    pauli_str = "I" * (n_qubits - 1 - j) + "Z" + "I" * j
    n_op_terms.append((pauli_str, -0.5))
N_op  = SparsePauliOp.from_list(n_op_terms)
N_mts = MajoranaTermSum.from_sparse_pauli_op(N_op)
print(f"N observable: {len(N_mts)} Majorana terms")

prop = MajoranaPropagator(
    UniformNoiseModel(damping=0.0),
    TruncationPolicy(weight_cutoff=None, coeff_cutoff=args.cutoff,
                     truncation_range=(None, 10_000_000)),
    n_threads=args.n_threads,
    progress_bar=True,
)

print("\nPropagating N through LUCJ circuit ...")
t0      = time.perf_counter()
UNU_mts = prop.propagate(N_mts, mc)
rt      = time.perf_counter() - t0

value = sum(c.real * m.trace_with_fock_state(0) for m, c in UNU_mts.items())
print(f"N_trunc = {value:.10e}")
print(f"N_exact = 54")

print("\nComputing ||[U^dag N U, N]|| ...")
comm = MajoranaTermSum()
for mi, ai in UNU_mts.items():
    for mj, bj in N_mts.items():
        if not mi.commutes_with(mj):
            phase, mij = mi @ mj
            comm.add(mij, 2 * ai * bj * phase)
comm_norm = np.sqrt(comm.norm_squared())
print(f"||[U^dag N U, N]|| = {comm_norm:.6e}")

os.makedirs("results", exist_ok=True)
out = "results/number-operator.npz"
np.savez(
    out,
    value           = np.float64(value),
    comm_norm       = np.float64(comm_norm),
    n_qubits        = np.int64(n_qubits),
    ccsd_energy     = np.float64(ccsd_energy),
    cutoff          = np.float64(args.cutoff),
    runtime_seconds = np.float64(rt),
)
print(f"Saved {out}")
