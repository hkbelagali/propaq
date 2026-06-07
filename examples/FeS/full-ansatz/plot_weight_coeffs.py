import numpy as np
import matplotlib.pyplot as plt

hamiltonian_cache = "compiled_hamiltonian_cache.npz"
weight = 5

data = np.load(hamiltonian_cache, allow_pickle=True)
paulis = data["paulis"]
coeffs = data["coeffs"]

def pauli_weight(label):
    return sum(1 for c in label if c != "I")

weights = np.array([pauli_weight(p) for p in paulis])
mask = weights == weight
w_coeffs = coeffs[mask]

print(f"Weight {weight}: {mask.sum()} terms")

abs_coeffs = np.abs(w_coeffs)
log_min = np.floor(np.log10(abs_coeffs.min()))
log_max = np.ceil(np.log10(abs_coeffs.max()))
bins = np.logspace(log_min, log_max, int(log_max - log_min) + 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(abs_coeffs, bins=bins)
ax.set_xscale("log")
ax.set_xlabel("|coeff|")
ax.set_ylabel("Count")
ax.set_title(f"Pauli coefficient magnitudes — weight {weight} terms ({mask.sum()} total)")
plt.tight_layout()
plt.savefig(f"coeffs_weight{weight}.png", dpi=150)
print(f"Saved coeffs_weight{weight}.png")
plt.show()
