"""
Plot propaq Ising benchmark results, mirroring the style of FIG. 7 in
the PauliPropagation.jl paper (arXiv:2505.21606).

Main panel: time per Trotter step vs step index for each thread count.
Inset    : Pauli term count vs step index (with truncation applied).

Usage:
    python plot_benchmark.py [--results results.npz] [--out benchmark.pdf]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

STYLE = Path(__file__).resolve().parent.parent / "presentation.mplstyle"

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Cyan → blue/purple → magenta, matching FIG. 7 colour ramp
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
    parser.add_argument("--results", default="results.npz", help="NPZ from run_benchmark.py")
    parser.add_argument("--out", default="benchmark.pdf", help="Output figure path")
    args = parser.parse_args()

    # if STYLE.exists():
        # plt.style.use(str(STYLE))

    data = np.load(args.results)
    thread_counts: list[int] = data["thread_counts"].tolist()
    n_steps = int(data["n_steps"])
    steps = np.arange(1, n_steps + 1)

    coeff_cutoff = float(data["coeff_cutoff"])
    J  = float(data.get("J",  1.0))
    HX = float(data.get("HX", 0.5))
    HZ = float(data.get("HZ", 0.5))
    DT = float(data.get("DT", 0.05))

    FS      = 20   # main axes: labels, ticks, legend, title
    FS_IN   = 20   # inset axes: labels and ticks

    fig, ax = plt.subplots(figsize=(8, 8))

    for n_threads in thread_counts:
        times = data[f"times_{n_threads}"]
        label = f"{n_threads} thread{'s' if n_threads > 1 else ' '}"
        ax.semilogy(steps, times, color=thread_color(n_threads), lw=1.3, label=label)

    ax.set_xlabel("Trotter Step", fontsize=FS)
    ax.set_ylabel("Time per Step [s]", fontsize=FS)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(1e-2, 1e4)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4])
    ax.tick_params(labelsize=FS)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, which="major", color="gray", linewidth=0.5, alpha=0.2)
    ax.legend(fontsize=FS-8, loc="lower right", framealpha=0.9)
    # ax.set_title(
    #     rf"6$\times$6 Ising  $J$={J}  $h_x$={HX}  $h_z$={HZ}  $dt$={DT}"
    #     rf"   cutoff $2^{{-20}}$",
    #     fontsize=FS,
    # )

    # ── Inset: Pauli term count vs step ───────────────────────────────────
    ax_in = ax.inset_axes([0.15, 0.68, 0.25, 0.25])
    ref = thread_counts[0]
    nterms = data[f"nterms_{ref}"]
    ax_in.semilogy(steps, nterms, color="k", lw=1.0)
    ax_in.set_xlabel("Trotter Step", fontsize=FS_IN)
    ax_in.set_ylabel("Paulis", fontsize=FS_IN)
    ax_in.set_xlim(0, n_steps)
    ax_in.set_xticks([0, 10, 20, 30])
    ax_in.set_yticks([1e0, 1e3, 1e6, 1e9])
    ax_in.grid(True, which="major", color="gray", linewidth=0.4, alpha=0.2)
    ax_in.tick_params(labelsize=FS_IN)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    print(f"Saved → {out}")

    # ── Summary table ──────────────────────────────────────────────────────
    has_evs = f"evs_{thread_counts[0]}" in data
    print(f"\n{'threads':>8}  {'step-1 [s]':>12}  {'step-30 [s]':>12}  "
          f"{'final terms':>12}  {'speedup vs 1T':>14}  {'final EV':>12}")
    t1 = data.get("times_1")
    for n_threads in thread_counts:
        times = data[f"times_{n_threads}"]
        nterms = data[f"nterms_{n_threads}"]
        speedup = f"{t1[-1] / times[-1]:.1f}x" if t1 is not None and n_threads != 1 else "—"
        ev_str = f"{data[f'evs_{n_threads}'][-1]:+.6f}" if has_evs else "—"
        print(f"{n_threads:>8}  {times[0]:>12.4f}  {times[-1]:>12.4f}  "
              f"{nterms[-1]:>12,}  {speedup:>14}  {ev_str:>12}")


if __name__ == "__main__":
    main()
