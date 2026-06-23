"""
Benchmark propaq's PauliPropagator on Trotterized 6x6 tilted-field Ising dynamics.

Reproduces the setup of FIG. 7 in the PauliPropagation.jl paper (arXiv:2505.21606):
  - 6x6 tilted-field Ising model (36 qubits)
  - Observable: Z_21 Z_22
  - dt = 0.05, 132 Pauli rotations per Trotter step (60 ZZ + 36 X + 36 Z)
  - Coefficient cutoff 2^{-20}, no weight truncation
  - Timing measured per Trotter step, accumulating the propagated state across steps
"""

import argparse
import time
from pathlib import Path

import numpy as np

from propaq.circuits import PauliCircuit, PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.noise import TruncationPolicy
from propaq.propagators import PauliPropagator

# ── Physical parameters ────────────────────────────────────────────────────
NROWS, NCOLS = 6, 6
N_QUBITS = NROWS * NCOLS  # 36
J = 1.0    # nearest-neighbour Ising coupling
HX = 0.5   # transverse (X) field
HZ = 0.5   # longitudinal (Z) field
DT = 0.05  # Trotter step size

# ── Benchmark defaults ─────────────────────────────────────────────────────
DEFAULT_N_STEPS = 30
DEFAULT_COEFF_CUTOFF = 2.0 ** (-20)   # ≈ 9.54e-7
DEFAULT_THREADS = [1, 2, 4, 8, 16, 32, 64]

# Observable qubits (0-indexed): Z_21 Z_22
OBS_QUBITS = (21, 22)


def build_trotter_step() -> PauliCircuit:
    """One first-order Trotter step for H = -J Σ ZZ - hx Σ X - hz Σ Z.

    propaq convention: PauliRotation(gen, angle) = exp(-i angle/2 * gen).
    The Heisenberg update is O → cos(angle)*O + sin(angle)*O' (paper Eq. 14).

    To implement exp(-i dt c G) = exp(-i (2c*dt)/2 * G), set angle = 2*c*dt.
    So for each Trotter factor:
        ZZ bond (coeff -J):  angle = 2*(-J)*dt = -2*J*dt
        X  site (coeff -hx): angle = 2*(-hx)*dt = -2*hx*dt
        Z  site (coeff -hz): angle = 2*(-hz)*dt = -2*hz*dt

    Bond count: 6*5 horizontal + 5*6 vertical = 60 ZZ bonds
    Single-qubit: 36 X + 36 Z = 72
    Total: 132 rotations per step.
    """
    rots = []
    # ZZ bonds — horizontal neighbours
    for r in range(NROWS):
        for c in range(NCOLS - 1):
            i = r * NCOLS + c
            j = i + 1
            gen = PauliString(0, (1 << i) | (1 << j), N_QUBITS)
            rots.append(PauliRotation(gen, -2 * J * DT))
    # ZZ bonds — vertical neighbours
    for r in range(NROWS - 1):
        for c in range(NCOLS):
            i = r * NCOLS + c
            j = i + NCOLS
            gen = PauliString(0, (1 << i) | (1 << j), N_QUBITS)
            rots.append(PauliRotation(gen, -2 * J * DT))
    # X terms
    for i in range(N_QUBITS):
        gen = PauliString(1 << i, 0, N_QUBITS)
        rots.append(PauliRotation(gen, -2 * HX * DT))
    # Z terms
    for i in range(N_QUBITS):
        gen = PauliString(0, 1 << i, N_QUBITS)
        rots.append(PauliRotation(gen, -2 * HZ * DT))

    n_rots = len(rots)
    assert n_rots == 132, f"Expected 132 rotations, got {n_rots}"
    return PauliCircuit(rots)


def build_observable() -> PauliTermSum:
    z_mask = sum(1 << q for q in OBS_QUBITS)
    return PauliTermSum({PauliString(0, z_mask, N_QUBITS): 1.0})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time per Trotter step for 6x6 Ising Pauli propagation"
    )
    parser.add_argument("--out", default="results.npz", help="Output .npz path")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument(
        "--threads", type=int, nargs="+", default=DEFAULT_THREADS,
        metavar="N", help="Thread counts to benchmark"
    )
    parser.add_argument(
        "--coeff-cutoff", type=float, default=DEFAULT_COEFF_CUTOFF,
        help="Coefficient truncation cutoff (default: 2^-20)"
    )
    args = parser.parse_args()

    trotter_step = build_trotter_step()
    print(f"Model: {NROWS}x{NCOLS} tilted-field Ising  J={J}  hx={HX}  hz={HZ}  dt={DT}")
    print(f"Circuit: {len(trotter_step.rotations)} rotations/step")
    print(f"Observable: Z_{OBS_QUBITS[0]} Z_{OBS_QUBITS[1]}")
    print(f"Truncation: coeff_cutoff={args.coeff_cutoff:.3e}, weight_cutoff=None")
    print(f"Steps: {args.n_steps}  |  Thread counts: {args.threads}\n")

    obs_init = build_observable()
    trunc = TruncationPolicy(weight_cutoff=None, coeff_cutoff=args.coeff_cutoff)

    all_times: dict[int, list[float]] = {}
    all_nterms: dict[int, list[int]] = {}
    all_evs: dict[int, list[float]] = {}

    for n_threads in args.threads:
        print(f"── n_threads={n_threads} " + "─" * 50)
        prop = PauliPropagator(noise=None, truncation=trunc, n_threads=n_threads)
        obs = obs_init  # propagate() copies internally, so reuse is safe
        step_times: list[float] = []
        step_nterms: list[int] = []
        step_evs: list[float] = []

        for step in range(args.n_steps):
            t0 = time.perf_counter()
            obs = prop.propagate(obs, trotter_step)
            elapsed = time.perf_counter() - t0

            # Expectation value ⟨0|O(t)|0⟩: sum coeff * ⟨0|P|0⟩ over all Pauli strings.
            # trace_with_fock_state returns 0 for any X/Y term, (-1)^popcount(z & state) for Z-only.
            ev = sum(c.real * term.trace_with_fock_state(0) for term, c in obs.items())

            step_times.append(elapsed)
            step_nterms.append(len(obs))
            step_evs.append(ev)
            print(f"  step {step + 1:2d}:  {len(obs):>12,} terms  {elapsed:.4f} s  ev={ev:+.8f}")

        all_times[n_threads] = step_times
        all_nterms[n_threads] = step_nterms
        all_evs[n_threads] = step_evs
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save: dict[str, object] = {
        "thread_counts": np.array(args.threads, dtype=np.int64),
        "n_steps": np.int64(args.n_steps),
        "coeff_cutoff": np.float64(args.coeff_cutoff),
        "J": np.float64(J),
        "HX": np.float64(HX),
        "HZ": np.float64(HZ),
        "DT": np.float64(DT),
        "obs_qubits": np.array(OBS_QUBITS, dtype=np.int64),
    }
    for n_threads in args.threads:
        save[f"times_{n_threads}"] = np.array(all_times[n_threads])
        save[f"nterms_{n_threads}"] = np.array(all_nterms[n_threads], dtype=np.int64)
        save[f"evs_{n_threads}"] = np.array(all_evs[n_threads])
    np.savez(str(out), **save)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
