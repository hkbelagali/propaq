"""
Variational optimization of LUCJ ansatz parameters.
"""

import argparse
import os
import pickle
import time
from math import floor, log10

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import ffsim
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp

from propaq.datatypes import MajoranaTermSum
from propaq.circuits import MajoranaCircuit
from propaq.propagators import MajoranaPropagator
from propaq.noise import UniformNoiseModel, TruncationPolicy

parser = argparse.ArgumentParser(description="Variational LUCJ optimization")
parser.add_argument("--cutoff",     type=float, default=1e-4)
parser.add_argument("--n-threads",  type=int,   default=64)
parser.add_argument("--batch-size", type=int,   default=10_000)
parser.add_argument("--optimizer",  type=str,   default="cobyla",
                    choices=["cobyla", "lbfgsb", "nelder-mead"])
parser.add_argument("--maxiter",    type=int,   default=500)
parser.add_argument("--rhobeg",     type=float, default=0.05)
parser.add_argument("--resume",     action="store_true")
args = parser.parse_args()

# ─── Load initialization ─────────────────────────────────────────────────────
init = np.load("optimization_init.npz", allow_pickle=False)
num_orb     = int(init["num_orb"])
num_elec_a  = int(init["num_elec_a"])
num_elec_b  = int(init["num_elec_b"])
n_reps      = int(init["n_reps"])
nelec       = (num_elec_a, num_elec_b)
ccsd_energy = float(init["ccsd_energy"])

alpha_alpha_indices = [tuple(int(v) for v in p) for p in init["alpha_alpha_indices"]]
alpha_beta_indices  = [tuple(int(v) for v in p) for p in init["alpha_beta_indices"]]

with open("ucj_op_init.pkl", "rb") as f:
    initial_ucj_op = pickle.load(f)

# Pull everything from the pickle — avoids any npz serialization mismatch.
initial_or = np.array(initial_ucj_op.orbital_rotations)
final_or   = getattr(initial_ucj_op, "final_orbital_rotation", None)
if final_or is not None:
    final_or = np.asarray(final_or)

print(f"orbital_rotations shape:    {initial_or.shape}  dtype: {initial_or.dtype}")
print(f"final_orbital_rotation:     {None if final_or is None else final_or.shape}")

n_aa        = len(alpha_alpha_indices)
n_ab        = len(alpha_beta_indices)
n_dc_params = n_aa + n_ab              # per rep
n_params    = n_reps * n_dc_params

print(f"System: {num_orb} orbitals, {num_elec_a}α/{num_elec_b}β, {n_reps} UCJ reps")
print(f"Orbital rotations: FROZEN at CCSD")
print(f"Diag-Coulomb params: {n_reps} × {n_dc_params} = {n_params}  "
      f"({n_aa} α-α + {n_ab} α-β pairs, sparse)")

initial_dc = np.array(initial_ucj_op.diag_coulomb_mats, dtype=float)
print(f"  diag_coulomb_mats shape: {initial_dc.shape}  dtype: {initial_dc.dtype}")


def unpack_dc(x_dc):
    """Flat delta params → (n_reps, 2, norb, norb) = CCSD base + sparse perturbation."""
    dc_mats = initial_dc.copy()
    for r in range(n_reps):
        base = r * n_dc_params
        for i, (p, q) in enumerate(alpha_alpha_indices):
            dc_mats[r, 0, p, q] += x_dc[base + i]
            dc_mats[r, 0, q, p] += x_dc[base + i]
        for i, (p, _) in enumerate(alpha_beta_indices):
            dc_mats[r, 1, p, p] += x_dc[base + n_aa + i]
    return dc_mats

coupling_map = CouplingMap.from_heavy_hex(distance=7)
backend = GenericBackendV2(
    coupling_map.size(),
    coupling_map=coupling_map,
    basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
)
pass_manager, _ = ffsim.qiskit.generate_lucj_pass_manager(
    backend=backend,
    norb=num_orb,
    connectivity="heavy-hex",
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    optimization_level=2,
)

prop = MajoranaPropagator(
    None,
    TruncationPolicy(weight_cutoff=None, coeff_cutoff=args.cutoff,
                     truncation_range=(None, 10_000_000)),
    n_threads=args.n_threads,
    progress_bar=True,
)

qubits = qiskit.QuantumRegister(2 * num_orb, name="q")

_ref_circ = qiskit.QuantumCircuit(qubits)
_ref_circ.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
_ref_circ.append(ffsim.qiskit.UCJOpSpinBalancedJW(initial_ucj_op), qubits)
_ref_compiled = pass_manager.run(_ref_circ)
print(f"\nReference compilation: {_ref_compiled.num_qubits} physical qubits")

_ham_logical_cache = np.load("../FeS-LUCJ/hamiltonian_cache.npz", allow_pickle=False)
_ham_logical = SparsePauliOp.from_list(
    list(zip(_ham_logical_cache["paulis"].astype(str), _ham_logical_cache["coeffs"]))
)
_ham_physical   = _ham_logical.apply_layout(_ref_compiled.layout)
print(f"Physical Hamiltonian:  {_ham_physical.num_qubits} qubits, "
      f"{len(_ham_physical)} terms")

coeffs_raw = np.real(np.array(_ham_physical.coeffs))
paulis_raw = np.array(_ham_physical.paulis.to_labels())
weights    = np.array([sum(ch != "I" for ch in p) for p in paulis_raw])

# ECORE: constant (identity-Pauli) shift added to every energy evaluation
ecore = float(coeffs_raw[weights == 0].sum())
print(f"ECORE (identity terms): {ecore:.10f} Ha")

EVAL_ORDERS = {0, -1, -2}
keep_mask   = np.array([
    w > 0 and abs(c) > 0 and floor(log10(abs(c))) in EVAL_ORDERS
    for c, w in zip(coeffs_raw, weights)
])
ham_trunc = SparsePauliOp.from_list(
    list(zip(paulis_raw[keep_mask], _ham_physical.coeffs[keep_mask]))
)
print(f"Truncated Hamiltonian: {keep_mask.sum()} / {len(keep_mask)} terms "
      f"(orders {sorted(EVAL_ORDERS, reverse=True)})")

obs     = MajoranaTermSum.from_sparse_pauli_op(ham_trunc)
items   = list(obs.items())
batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]
print(f"Majorana monomials: {len(items)} in {len(batches)} batch(es)")

def compute_energy(x):
    dc_mats = unpack_dc(x)
    ucj_op  = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=dc_mats,
        orbital_rotations=initial_or,       # frozen at CCSD
        final_orbital_rotation=final_or,
    )
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    compiled = pass_manager.run(circuit)
    mc = MajoranaCircuit.from_qiskit(compiled.copy(), n_modes=2 * compiled.num_qubits)

    total = ecore
    for batch in batches:
        ts = MajoranaTermSum()
        for monomial, coeff in batch:
            ts.add(monomial, coeff)
        total += float(np.real(
            prop.expectation_value(ts, mc, initial_state=0).expectation_value
        ))
    return total

print("\nBuilding initial parameter vector...")
# All zeros → unpack_dc gives exactly initial_dc (CCSD base, no perturbation)
x0 = np.zeros(n_params, dtype=float)

os.makedirs("results", exist_ok=True)
checkpoint_file = "results/opt_checkpoint.npz"

energy_hist = []
param_hist  = []
time_hist   = []
n_evals     = [0]

if args.resume and os.path.exists(checkpoint_file):
    ckpt = np.load(checkpoint_file, allow_pickle=False)
    x0          = ckpt["x_last"]
    energy_hist = list(ckpt["energy_history"])
    param_hist  = list(ckpt["param_history"])
    n_evals[0]  = int(ckpt["n_evals"])
    print(f"Resumed: {n_evals[0]} prior evals, best E = {min(energy_hist):.10f}")

print("\nEvaluating initial energy (x=0 = CCSD)...")
e0 = compute_energy(x0)
print(f"  E_init (CCSD) = {e0:.10f}")

opt_method  = {"cobyla": "COBYLA", "lbfgsb": "L-BFGS-B",
               "nelder-mead": "Nelder-Mead"}[args.optimizer]
opt_options: dict = {"maxiter": args.maxiter}
if args.optimizer == "cobyla":
    opt_options["rhobeg"] = args.rhobeg
elif args.optimizer == "lbfgsb":
    opt_options.update({"ftol": 1e-6, "gtol": 1e-4, "eps": 1e-3})

print(f"\nRunning {opt_method} on {n_params} parameters "
      f"(max {args.maxiter} iter, cutoff={args.cutoff})...")
t_start = time.perf_counter()

pbar = tqdm(
    total=args.maxiter,
    initial=n_evals[0],
    desc=opt_method,
    unit="eval",
    dynamic_ncols=True,
)

def objective(x):
    t0 = time.perf_counter()
    e  = compute_energy(x)
    dt = time.perf_counter() - t0
    n_evals[0] += 1
    energy_hist.append(e)
    param_hist.append(x.copy())
    time_hist.append(dt)
    best = min(energy_hist)
    pbar.update(1)
    pbar.set_postfix(
        E=f"{e:.6f}",
        best=f"{best:.6f}",
        dCCSD=f"{best - ccsd_energy:+.4f}",
        t=f"{dt:.1f}s",
    )
    np.savez(
        checkpoint_file,
        x_last=x,
        energy_history=np.array(energy_hist),
        param_history=np.array(param_hist),
        time_history=np.array(time_hist),
        n_evals=np.int64(n_evals[0]),
        ccsd_energy=np.float64(ccsd_energy),
    )
    return e

opt_result = minimize(objective, x0, method=opt_method, options=opt_options)

pbar.close()
t_total     = time.perf_counter() - t_start
best_energy = min(energy_hist)
best_idx    = int(np.argmin(energy_hist))
x_best      = param_hist[best_idx]

print(f"\nOptimization complete: {n_evals[0]} evaluations in {t_total / 60:.1f} min")
print(f"  Best E_trunc  = {best_energy:.10f}")
print(f"  E_init        = {e0:.10f}")
print(f"  E_ccsd        = {ccsd_energy:.10f}")
print(f"  ΔE (best−init)= {best_energy - e0:+.6f} Ha")

# ─── Save results ────────────────────────────────────────────────────────────
dc_opt = unpack_dc(x_best)
np.savez(
    "results/variational_result.npz",
    x_best=x_best,
    optimal_orbital_rotations=initial_or,   # frozen CCSD
    optimal_diag_coulomb_mats=dc_opt,
    ecore=np.float64(ecore),
    energy_history=np.array(energy_hist),
    time_history=np.array(time_hist),
    n_evals=np.int64(n_evals[0]),
    best_energy=np.float64(best_energy),
    e_init=np.float64(e0),
    ccsd_energy=np.float64(ccsd_energy),
    optimizer=np.bytes_(args.optimizer),
    eval_orders=np.array(sorted(EVAL_ORDERS, reverse=True)),
    total_time_seconds=np.float64(t_total),
    coeff_cutoff=np.float64(args.cutoff),
)
print("Saved results/variational_result.npz")

print("\nRecompiling optimal circuit (optimization_level=3)...")
pm_final, _ = ffsim.qiskit.generate_lucj_pass_manager(
    backend=backend,
    norb=num_orb,
    connectivity="heavy-hex",
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    optimization_level=3,
)
ucj_opt     = ffsim.UCJOpSpinBalanced(
    diag_coulomb_mats=dc_opt,
    orbital_rotations=initial_or,
    final_orbital_rotation=final_or,
)
circuit_opt = qiskit.QuantumCircuit(qubits)
circuit_opt.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
circuit_opt.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_opt), qubits)
compiled_opt = pm_final.run(circuit_opt)

from qiskit import qpy as _qpy
with open("results/FeS_LUCJ_optimal_circuit.qpy", "wb") as f:
    _qpy.dump(compiled_opt, f)
print("Saved results/FeS_LUCJ_optimal_circuit.qpy")

if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"Checkpoint removed: {checkpoint_file}")
