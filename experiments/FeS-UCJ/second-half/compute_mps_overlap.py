from propaq.datatypes import MajoranaTermSum, MajoranaTermStreamer, mps_pauli_overlap_sum
import quimb as qu
import numpy as np

forward_mps, _ = qu.load_from_disk("mps_forward_FeS_LUCJ_1.5_layers_final_op_index_1422")
L = forward_mps.L

# pull out the tensors to avoid Python overhead, as contiguous arrays so that they can be passed to Rust
tensors = [np.ascontiguousarray(t.data, dtype=np.complex128) for t in forward_mps]

# just a sanity check to make sure that || |mps> ||^2 = <mps|I|mps>

# norm_sq_quimb = forward_mps.norm() ** 2
# norm_sq_rust = mps_pauli_overlap_sum(tensors, [("I" * L, 1.0 + 0j)])
# rel_err = abs(norm_sq_rust - norm_sq_quimb) / max(abs(norm_sq_quimb), 1e-300)
# if rel_err > 1e-8:
#     raise RuntimeError(
#         f"norm check failed: quimb={norm_sq_quimb:.10f}, rust={norm_sq_rust:.10f}, rel_err={rel_err:.2e}"
#     )
# print(f"Sanity check passed: <MPS|I|MPS> = {norm_sq_rust.real:.10f}  (quimb norm² = {norm_sq_quimb:.10f})", flush=True)

# this lazily loads the Majorana terms from disk
streamer = MajoranaTermStreamer.from_file(
    "results/FeS_LUCJ_o-2_-1_0_00000of00001cutoff1.0e-04_b00000.gz"
)
print("Opened MajoranaTermStreamer: results/FeS_LUCJ_o-2_-1_0_00000of00001cutoff1.0e-04_b00000.gz")

BATCH_SIZE = 1

ev: complex = 0
batch: list[tuple[str, complex]] = []


# processes a batch of terms and adds their contributions to the expectation value
def flush_batch() -> None:
    print(f"\nFlushing batch of {len(batch)} terms...", flush=True)
    global ev
    if batch:
        ev += mps_pauli_overlap_sum(tensors, batch)
        batch.clear()


for i, (monomial, coeff) in enumerate(streamer):
    # quick way to Majorana monomials as Pauli strings
    ts = MajoranaTermSum()
    ts.add(monomial, coeff)  # type: ignore[arg-type]
    spo = ts.to_sparse_pauli_op()  # type: ignore[attr-defined]

    for pauli_str, pauli_coeff in spo.to_list():
        batch.append((pauli_str, complex(pauli_coeff)))

    if len(batch) >= BATCH_SIZE:
        flush_batch()

    if i % BATCH_SIZE == 0:
        print(f"\r[{i}] ev = {ev.real:.10e}", end="", flush=True)

flush_batch()
print(f"\nFinal expectation value: {ev.real:.10e}")
