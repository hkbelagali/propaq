"""
Plot expectation value of Z_21 Z_22 vs time from run_benchmark.py results.

Usage:
    python plot_ev.py [--results results.npz] [--out ev.pdf]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE = Path(__file__).resolve().parent.parent / "presentation.mplstyle"

_THREAD_COLORS = {
    1:  "#00c8ff",
    2:  "#4499ff",
    4:  "#6655ff",
    8:  "#9922ee",
    16: "#cc00ff",
    32: "#e600bb",
    64: "#ff0099",
}


def thread_color(n: int) -> str:
    return _THREAD_COLORS.get(n, "#888888")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.npz")
    parser.add_argument("--out", default="ev.pdf")
    args = parser.parse_args()

    if STYLE.exists():
        plt.style.use(str(STYLE))

    data = np.load(args.results)
    thread_counts: list[int] = data["thread_counts"].tolist()
    n_steps = int(data["n_steps"])
    steps = np.arange(1, n_steps + 1)
    DT = float(data.get("DT", 0.05))
    J  = float(data.get("J",  1.0))
    HX = float(data.get("HX", 0.5))
    HZ = float(data.get("HZ", 0.5))

    if f"evs_{thread_counts[0]}" not in data:
        raise SystemExit("No EV data found — re-run run_benchmark.py first.")

    # Prepend t=0: ⟨0|Z_21 Z_22|0⟩ = +1 exactly
    t_vals = np.concatenate([[0.0], steps * DT])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # All thread counts must produce identical EVs; overlay as sanity check
    for n_threads in thread_counts:
        evs = np.concatenate([[1.0], data[f"evs_{n_threads}"]])
        label = f"{n_threads} thread{'s' if n_threads > 1 else ' '}"
        ax.plot(t_vals, evs, color=thread_color(n_threads), lw=1.3, label=label)

    obs_qubits = data.get("obs_qubits", np.array([21, 22])).tolist()
    obs_label = r"Z_{%d}Z_{%d}" % tuple(obs_qubits)

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"Time  $t = n_\mathrm{step} \cdot dt$")
    ax.set_ylabel(rf"$\langle {obs_label} \rangle(t)$")
    ax.set_title(
        rf"6$\times$6 Ising  $J$={J}  $h_x$={HX}  $h_z$={HZ}  $dt$={DT}"
        rf"   cutoff $2^{{-20}}$",
        fontsize=9,
    )
    ax.set_xlim(0, t_vals[-1])
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
