"""
Direct number-projection numerator via N-sandwich expectation values.

For each Majorana string M (coefficient c) streamed from the propagated
Hamiltonian gz files, expands inline:

    N · M · N  =  Σ_{j,k}  n_j · M · n_k,   n_j = (I + M_j)/2

Each product monomial is filtered to those with exactly n_orb active Majorana
modes in the spin-up sector AND n_orb in the spin-down sector (sector
assignment derived from the compiled circuit's physical layout), then its
vacuum-Fock-state trace is taken and accumulated immediately:

    numerator  +=  Re(c_product) · tr(m_product, fock_state=0)

No intermediate hashmap is built.  Terms are streamed, expanded, filtered,
traced, and discarded.  A snapshot of the cumulative EV is recorded at the
end of each gz file and saved to results/cumulative_ev.npz.

ffsim's UCJOpSpinBalancedJW uses interleaved Jordan-Wigner ordering:
  logical qubit 2p   → α (spin-up)  orbital p
  logical qubit 2p+1 → β (spin-down) orbital p

After pass_manager.run() maps the circuit to a heavy-hex backend and
apply_layout() re-indexes the Hamiltonian to physical qubits, the α and β
qubits are scattered across the device.  The compiled .qpy circuit's layout
is used here to recover which physical qubit indices are spin-up vs spin-down.

The projected energy is E = numerator / denominator (denominator from
build_circuit.py).
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from qiskit import qpy
from tqdm import tqdm

from propaq._rust_core import MajoranaMonomial, MajoranaTermStreamer
from propaq.datatypes._abstract import BitMask

# --------------------------------------------------------------------------- #
# Worker-process globals                                                        #
# --------------------------------------------------------------------------- #
_N_ITEMS: list[tuple[MajoranaMonomial, float]] = []
_FOCK_STATE: int = 0
_N_MODES: int = 0
_N_UP: int = 0
_N_DOWN: int = 0
_SPIN_UP_MASK: int = 0
_SPIN_DOWN_MASK: int = 0


def _worker_init(
    n_modes: int,
    n_qubits: int,
    fock_state: int,
    alpha_phys: tuple[int, ...],
    beta_phys: tuple[int, ...],
) -> None:
    global _N_ITEMS, _FOCK_STATE, _N_MODES, _N_UP, _N_DOWN, _SPIN_UP_MASK, _SPIN_DOWN_MASK
    _N_MODES    = n_modes
    _FOCK_STATE = fock_state
    _N_UP       = len(alpha_phys)
    _N_DOWN     = len(beta_phys)
    # Physical qubit q contributes Majorana modes at bits 2q and 2q+1.
    _SPIN_UP_MASK   = sum(3 << (2 * q) for q in alpha_phys)
    _SPIN_DOWN_MASK = sum(3 << (2 * q) for q in beta_phys)

    identity = MajoranaMonomial(BitMask(0), n_modes, is_number_preserving=True)
    items: list[tuple[MajoranaMonomial, float]] = [(identity, float(n_qubits) / 2.0)]
    for q in range(n_qubits):
        modes_bits = (1 << (2 * q)) | (1 << (2 * q + 1))
        items.append(
            (MajoranaMonomial(BitMask(modes_bits), n_modes, is_number_preserving=True), 0.5)
        )
    _N_ITEMS = items


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _chunk_ev(chunk: list[tuple[int, complex, bool]]) -> float:
    """
    Expand N·M·N for every term in chunk, filter each product monomial to
    those with exactly _N_UP spin-up and _N_DOWN spin-down active Majorana
    modes, and return their cumulative vacuum-Fock-state trace contribution.
    No intermediate collection is built.
    """
    partial_ev = 0.0
    for modes, coeff, is_np in chunk:
        m = MajoranaMonomial(modes, _N_MODES, is_number_preserving=is_np)
        for m_left, c_left in _N_ITEMS:
            phase_l, m_lm = m_left @ m
            for m_right, c_right in _N_ITEMS:
                phase_r, m_lmr = m_lm @ m_right
                modes_int = int(m_lmr.modes)
                if (
                    _popcount(modes_int & _SPIN_UP_MASK)   != _N_UP
                    or _popcount(modes_int & _SPIN_DOWN_MASK) != _N_DOWN
                ):
                    continue
                c_product = coeff.real * c_left * c_right * phase_l * phase_r
                partial_ev += c_product * m_lmr.trace_with_fock_state(_FOCK_STATE)
    return partial_ev


def _split(lst: list, n: int) -> list[list]:
    k, r = divmod(len(lst), n)
    out, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        if start < end:
            out.append(lst[start:end])
        start = end
    return out


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--fock-state",   type=int,  default=0,
                    help="Reference Fock state integer (default: 0 = vacuum)")
parser.add_argument("--n-workers",    type=int,  default=128,
                    help="Worker processes")
parser.add_argument("--batch-size",   type=int,  default=10_000_000,
                    help="Terms streamed before dispatching to workers")
parser.add_argument("--ham-results",  type=str,  default="../hamiltonian/results",
                    help="Directory containing *.gz files from hamiltonian/run_LUCJ.py")
parser.add_argument("--denom-file",   type=str,  default="results/denominator.npz",
                    help="npz produced by build_circuit.py")
parser.add_argument("--circuit-file", type=str,  default="../FeS_LUCJ_circuit.qpy",
                    help="compiled .qpy circuit from build_LUCJ.py (used to extract spin-sector layout)")
args = parser.parse_args()


def _extract_spin_layout(
    circuit_path: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Load the compiled circuit and return the physical qubit indices for the
    α (spin-up) and β (spin-down) orbitals.

    ffsim's UCJOpSpinBalancedJW uses interleaved Jordan-Wigner ordering:
      logical qubit 2p   → α orbital p
      logical qubit 2p+1 → β orbital p

    final_index_layout(filter_ancillas=True) returns one entry per original
    (pre-transpile) qubit: phys[i] = physical qubit for logical qubit i.
    Its length is 2*n_orb, NOT compiled.num_qubits (the full device size).
    """
    with open(circuit_path, "rb") as f:
        compiled = qpy.load(f)[0]

    phys  = compiled.layout.final_index_layout(filter_ancillas=True)
    n_orb = len(phys) // 2

    alpha_phys = tuple(phys[2 * p]     for p in range(n_orb))
    beta_phys  = tuple(phys[2 * p + 1] for p in range(n_orb))
    return alpha_phys, beta_phys


def main() -> None:
    denom_data  = np.load(args.denom_file, allow_pickle=False)
    denominator = float(denom_data["denominator"])
    n_qubits    = int(denom_data["n_qubits"])
    ccsd_energy = float(denom_data["ccsd_energy"])
    n_modes     = 2 * n_qubits
    fock_state  = args.fock_state

    alpha_phys, beta_phys = _extract_spin_layout(args.circuit_file)
    n_orb = len(alpha_phys)

    print(f"n_qubits     = {n_qubits}  (physical, incl. routing qubits)")
    print(f"n_orb        = {n_orb}  (α physical qubits: {sorted(alpha_phys)[:4]}...)")
    print(f"fock_state   = {fock_state}")
    print(f"n_workers    = {args.n_workers}")
    print(f"batch_size   = {args.batch_size:,}")
    print(f"Denominator  = {denominator:.10e}")
    print(f"CCSD energy  = {ccsd_energy:.10e}")

    ham_dir  = Path(args.ham_results)
    gz_files = sorted(ham_dir.glob("*.gz"))
    print(f"\nFound {len(gz_files)} gz file(s) in {ham_dir}")
    if not gz_files:
        raise FileNotFoundError(f"No *.gz files in {ham_dir}. Run hamiltonian/run_LUCJ.py first.")

    os.makedirs("results", exist_ok=True)

    term_count        = 0
    running_numerator = 0.0
    cum_terms:     list[int]   = []
    cum_numerator: list[float] = []
    cum_ev:        list[float] = []
    batch: list[tuple[int, complex, bool]] = []

    with mp.Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(n_modes, n_qubits, fock_state, alpha_phys, beta_phys),
    ) as pool:

        def flush(label: str) -> None:
            nonlocal term_count, running_numerator
            chunks = _split(batch, args.n_workers)
            for partial_ev in tqdm(
                pool.imap(_chunk_ev, chunks),
                total=len(chunks),
                desc=label,
                unit="chunk",
            ):
                running_numerator += partial_ev
            term_count += len(batch)
            batch.clear()

        for gz_path in gz_files:
            print(f"\nStreaming {gz_path.name} ...", flush=True)
            for monomial, coeff in MajoranaTermStreamer.from_file(str(gz_path)):
                batch.append((monomial.modes, coeff, monomial.is_number_preserving))
                if len(batch) >= args.batch_size:
                    flush(gz_path.stem)

            if batch:
                flush(gz_path.stem)

            current_ev = running_numerator / denominator
            cum_terms.append(term_count)
            cum_numerator.append(running_numerator)
            cum_ev.append(current_ev)
            print(
                f"\n[{gz_path.name}]  cumulative E = {current_ev:.10e}"
                f"  ({term_count:,} terms streamed)"
            )

    E_projected = running_numerator / denominator

    print(f"\nNumerator  <psi|NHN|psi>  = {running_numerator:.10e}")
    print(f"Denominator <psi|N|psi>   = {denominator:.10e}")
    print(f"Projected energy E        = {E_projected:.10e}")
    print(f"CCSD energy               = {ccsd_energy:.10e}")
    print(f"Difference (E - CCSD)     = {E_projected - ccsd_energy:.10e}")
    print(f"Total terms processed     = {term_count:,}")

    out = "results/projected_energy.npz"
    np.savez(
        out,
        numerator        = np.float64(running_numerator),
        denominator      = np.float64(denominator),
        projected_energy = np.float64(E_projected),
        ccsd_energy      = np.float64(ccsd_energy),
        fock_state       = np.int64(fock_state),
        n_qubits         = np.int64(n_qubits),
        n_terms          = np.int64(term_count),
    )
    print(f"Saved {out}")

    cum_out = "results/cumulative_ev.npz"
    np.savez(
        cum_out,
        terms     = np.array(cum_terms,     dtype=np.int64),
        numerator = np.array(cum_numerator, dtype=np.float64),
        ev        = np.array(cum_ev,        dtype=np.float64),
    )
    print(f"Saved {cum_out}")


if __name__ == "__main__":
    main()
