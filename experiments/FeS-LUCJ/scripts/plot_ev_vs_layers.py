"""
Plot total expectation value vs number of LUCJ layers for FeS runs.

Data sources (noiseless):
  0.5, 2.0, 2.5 layers — sum ev_sum from refined_data/ + ECORE
  3.0 layers           — sum 1e-6-cutoff values from ZCE npz files + ECORE
Data sources (noisy):
  1.0 layers           — sum ev_sum from weight-based refined_data/ + ECORE
"""

from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parents[2] / "presentation.mplstyle")
import numpy as np

BASE = Path(__file__).resolve().parents[1]

CCSD_ENERGY = float(np.loadtxt(BASE / "energy_ccsd.txt"))
HF_ENERGY   = float(np.loadtxt(BASE / "energy_hf.txt"))


def ecore_from_cache(cache_path: Path) -> float:
    c = np.load(cache_path, allow_pickle=False)
    paulis = c["paulis"].astype(str)
    coeffs = c["coeffs"].real
    pw = np.array([sum(ch != "I" for ch in s) for s in paulis])
    return float(coeffs[pw == 0].sum())


def ev_from_refined(layer_dir: Path, pattern: str = "FeS_LUCJ_o*.npz") -> float:
    ecore = ecore_from_cache(layer_dir / "compiled_hamiltonian_cache.npz")
    total = ecore
    for f in (layer_dir / "refined_data").glob(pattern):
        d = np.load(f, allow_pickle=False)
        total += float(d["ev_sum"])
    return total


def ev_from_zce(layer_dir: Path, cutoff: float = 1e-6) -> float:
    ecore = ecore_from_cache(layer_dir / "compiled_hamiltonian_cache.npz")
    total = ecore
    for f in sorted((layer_dir / "results").glob("FeS_LUCJ_zce_o*.npz")):
        d = np.load(f, allow_pickle=False)
        cutoffs = np.asarray(d["coeff_cutoff_values"], dtype=float)
        vals    = np.asarray(d["summed_coeff_values"],  dtype=float)
        idx = np.argmin(np.abs(cutoffs - cutoff))
        total += float(vals[idx])
    return total


noiseless_ev = {
    1.5: ev_from_refined(BASE / "one-and-half-layer-noiseless"),
    2.0: ev_from_refined(BASE / "two-layer-noiseless"),
    2.5: ev_from_refined(BASE / "two-and-half-layer-noiseless"),
    3.0: ev_from_zce(BASE / "three-layer-noiseless", cutoff=1e-6),
}
noisy_ev = {
    1.0: ev_from_refined(BASE / "one-layer-noisy", pattern="FeS_LUCJ_w*.npz"),
}

for n, ev in sorted(noiseless_ev.items()):
    print(f"noiseless {n} layers: EV = {ev:.6f}")
for n, ev in sorted(noisy_ev.items()):
    print(f"noisy     {n} layers: EV = {ev:.6f}")
print(f"HF:   {HF_ENERGY:.6f}")
print(f"CCSD: {CCSD_ENERGY:.6f}")

nl_xs = sorted(noiseless_ev)
nl_ys = [noiseless_ev[x] for x in nl_xs]
ny_xs = sorted(noisy_ev)
ny_ys = [noisy_ev[x] for x in ny_xs]

all_xs = sorted(set(nl_xs) | set(ny_xs))

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(nl_xs, nl_ys, "o-", color="steelblue", lw=2, ms=7, zorder=3, label="Noiseless")
offsets = {2.0: (0, 18), 3.0: (22, 9)}
for x, y in zip(nl_xs, nl_ys):
    dx, dy = offsets.get(x, (0, 9))
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha="center", fontsize=9, color="steelblue")

ax.plot(ny_xs, ny_ys, "^", color="tomato", ms=9, zorder=3, label="Noisy")
for x, y in zip(ny_xs, ny_ys):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color="tomato")

ax.axhline(HF_ENERGY,   color="tab:orange", lw=1.5, ls="--",
           label=f"HF ({HF_ENERGY:.4f})")
ax.axhline(CCSD_ENERGY, color="tab:green",  lw=1.5, ls="--",
           label=f"CCSD ({CCSD_ENERGY:.4f})")

ax.set_xticks(all_xs)
ax.set_xticklabels([str(x) for x in all_xs])
ax.set_xlabel("Number of LUCJ layers")
ax.set_ylabel(r"$\langle H \rangle$ (Hartree)")
ax.set_title("FeS LUCJ: expectation value vs layers")
ax.legend(fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
out = BASE / "plots" / "ev_vs_layers.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved {out}")
