import os

N_WORKERS        = int(os.environ.get("N_WORKERS", 128))
BLAS_THREADS     = int(os.environ.get("BLAS_THREADS", 1))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", 50_000))
os.environ["OMP_NUM_THREADS"]     = str(BLAS_THREADS)
os.environ["MKL_NUM_THREADS"]     = str(BLAS_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]= str(BLAS_THREADS)

import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import quimb as qu

from propaq.datatypes import MajoranaTermSum, MajoranaTermStreamer

STREAMER_FILE   = "results/FeS_LUCJ_o-2_-1_0_00000of00001cutoff1.0e-04_b00000.gz"
CHECKPOINT_FILE = "checkpoint.json"
REPORT_EVERY    = 10_000

forward_mps, _ = qu.load_from_disk("mps_forward_FeS_LUCJ_1.5_layers_final_op_index_1422")
MPS_L = forward_mps.L

sites: list[np.ndarray] = []
for i, t in enumerate(forward_mps):
    A = np.ascontiguousarray(t.data, dtype=np.complex128)
    if A.ndim == 2:
        d = A.shape[0]
        A = A.reshape(1, d, 2) if i == 0 else A.reshape(d, 1, 2)
    sites.append(A)

def pauli_expectation(pauli_str: str) -> complex:
    env = np.ones((1, 1), dtype=np.complex128)
    for site, ch in zip(sites, reversed(pauli_str)):
        A0 = site[:, :, 0]
        A1 = site[:, :, 1]
        if   ch == 'X': k0, k1 = A1,        A0
        elif ch == 'Y': k0, k1 = -1j * A1,  1j * A0
        elif ch == 'Z': k0, k1 = A0,        -A1
        else:            k0, k1 = A0,         A1
        env = A0.conj().T @ env @ k0 + A1.conj().T @ env @ k1
    return complex(env[0, 0])

norm_sq_quimb = forward_mps.norm() ** 2
norm_sq_np    = pauli_expectation("I" * MPS_L).real
rel_err = abs(norm_sq_np - norm_sq_quimb) / max(abs(norm_sq_quimb), 1e-300)
if rel_err > 1e-8:
    raise RuntimeError(
        f"norm check failed: quimb={norm_sq_quimb:.10f}, numpy={norm_sq_np:.10f}, "
        f"rel_err={rel_err:.2e}"
    )
print(f"Sanity check passed: <MPS|I|MPS> = {norm_sq_np:.10f}  "
      f"(quimb norm² = {norm_sq_quimb:.10f})", flush=True)

def load_checkpoint() -> tuple[int, complex]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ck = json.load(f)
        print(f"Resuming from checkpoint: {ck['n_done']} terms done, "
              f"ev = {ck['ev_real']:.10e}", flush=True)
        return ck['n_done'], complex(ck['ev_real'], ck['ev_imag'])
    return 0, complex(0.0)

def save_checkpoint(n_done: int, ev_total: complex) -> None:
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({'n_done': n_done, 'ev_real': ev_total.real, 'ev_imag': ev_total.imag}, f)
    os.replace(tmp, CHECKPOINT_FILE)
    print(f"\n[checkpoint] {n_done} done, ev = {ev_total.real:.10e}", flush=True)

def monomial_to_pauli(monomial, coeff: complex) -> tuple[str, complex]:
    ts = MajoranaTermSum()
    ts.add(monomial, coeff)
    [(ps, pc)] = ts.to_sparse_pauli_op().to_list()
    return ps, complex(pc)

n_done, ev_total = load_checkpoint()
n_submitted = n_done

_stop = threading.Event()
def _sigint(sig, frame):
    _stop.set()
signal.signal(signal.SIGINT, _sigint)

streamer = MajoranaTermStreamer.from_file(STREAMER_FILE)

if n_done > 0:
    print(f"Fast-skipping {n_done} already-processed monomials ...", flush=True)
    for _ in zip(range(n_done), streamer):
        pass
    print("Done skipping.", flush=True)

print(f"Evaluating with {N_WORKERS} workers, {BLAS_THREADS} BLAS thread(s).", flush=True)

next_checkpoint = n_done + CHECKPOINT_EVERY

def _evaluate_term(pauli_str: str, coeff: complex) -> complex:
    return coeff * pauli_expectation(pauli_str)

with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures: dict = {}

    for monomial, coeff in streamer:
        if _stop.is_set():
            break

        ps, pauli_coeff = monomial_to_pauli(monomial, coeff)
        fut = pool.submit(_evaluate_term, ps, pauli_coeff)
        futures[fut] = None
        n_submitted += 1

        # Drain completed futures to bound memory.
        if len(futures) >= N_WORKERS * 4:
            done_futs = [f for f in futures if f.done()]
            for f in done_futs:
                ev_total += f.result()
                n_done += 1
                del futures[f]

        if n_submitted % REPORT_EVERY == 0:
            print(f"\r[{n_submitted:>9d} submitted | {n_done:>9d} done] "
                  f"ev = {ev_total.real:.10e}", end="", flush=True)

        if n_done >= next_checkpoint:
            for f in as_completed(list(futures)):
                ev_total += f.result()
                n_done += 1
            futures.clear()
            save_checkpoint(n_done, ev_total)
            next_checkpoint = n_done + CHECKPOINT_EVERY

    for f in as_completed(futures):
        ev_total += f.result()
        n_done += 1
        if n_done % REPORT_EVERY == 0:
            print(f"\r[draining: {n_done:>9d} done] ev = {ev_total.real:.10e}",
                  end="", flush=True)

save_checkpoint(n_done, ev_total)
print(f"\nFinal expectation value: {ev_total.real:.10e}")
print(f"Terms evaluated: {n_done}")
