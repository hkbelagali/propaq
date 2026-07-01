#!/mnt/home/belagal1/.conda/envs/env/bin/python3
"""
Plot map_terms and outbox_terms from a propagation log file.
Loops and redraws in-place. Ctrl-C to exit.

Usage:
  python3 plot_terms.py                          # auto-picks newest .jsonl in results/
  python3 plot_terms.py results/FeS_LUCJ_....jsonl
  python3 plot_terms.py --last 500 --interval 3
"""

import argparse
import json
import math
import os
import sys
import time

import plotext as plt

parser = argparse.ArgumentParser()
parser.add_argument("logfile", nargs="?", default=None,
                    help="Path to .jsonl log file. Defaults to the most recently modified one in results/.")
parser.add_argument("--last", type=int, default=None,
                    help="Only show the last N gate events.")
parser.add_argument("--interval", type=float, default=2.0,
                    help="Seconds between refreshes (default: 2).")
args = parser.parse_args()

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

_SUFFIXES = {0: "", 3: "K", 6: "M", 9: "G"}

def _fmt(n):
    for exp in (9, 6, 3, 0):
        if n >= 10**exp:
            v = n / 10**exp
            return f"{v:.0f}{_SUFFIXES[exp]}" if v == int(v) else f"{v:.1f}{_SUFFIXES[exp]}"
    return str(n)


def resolve_path():
    if args.logfile:
        return args.logfile
    jsonl_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".jsonl")
    ]
    if not jsonl_files:
        print("No .jsonl files found in results/")
        sys.exit(1)
    return max(jsonl_files, key=os.path.getmtime)


def read_events(path):
    gate_events = []
    truncation_events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = entry.get("event")
            if ev == "gate":
                gate_events.append(entry)
            elif ev == "truncation":
                truncation_events.append(entry)
    return gate_events, truncation_events


def draw(path, gate_events, truncation_events):
    if args.last is not None:
        gate_events = gate_events[-args.last:]

    xs        = [e["gate_idx"]     for e in gate_events]
    map_terms = [e["map_terms"]    for e in gate_events]
    outbox    = [e["outbox_terms"] for e in gate_events]

    # thin if dense (keep at most ~600 pts for readability)
    n = len(xs)
    step = max(1, n // 600)
    xs        = xs[::step]
    map_terms = map_terms[::step]
    outbox    = outbox[::step]

    # manual log10 transform so we control tick placement
    def safe_log(vals):
        return [math.log10(max(v, 1)) for v in vals]

    xs_m = safe_log(map_terms)
    xs_o = safe_log(outbox)

    all_vals = map_terms + outbox
    lo = max(0, int(math.floor(math.log10(max(1, min(all_vals))))))
    hi = int(math.ceil(math.log10(max(all_vals))))
    tick_vals  = list(range(lo, hi + 1))
    tick_labels = [_fmt(10**p) for p in tick_vals]

    trunc_count = len([e for e in truncation_events if xs[0] <= e["gate_idx"] <= xs[-1]])

    last = gate_events[-1]
    w = plt.tw()

    plt.clf()
    plt.theme("dark")
    plt.plot_size(w, 30)
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("white")

    plt.plot(xs, xs_m, label="hashmap", color=(80, 200, 255))
    plt.plot(xs, xs_o, label="outbox",  color=(255, 200, 60))

    plt.yticks(tick_vals, tick_labels)
    plt.yfrequency(len(tick_vals))

    plt.title(
        f"gate {last['gate_idx']} / layer {last['layer_idx']}"
        f"   map={_fmt(last['map_terms'])}   outbox={_fmt(last['outbox_terms'])}"
        + (f"   {trunc_count} truncations" if trunc_count else "")
    )
    plt.xlabel("gate index")
    plt.ylabel("terms (log scale)")
    plt.show()


path = resolve_path()
print(f"Watching {os.path.basename(path)}  (Ctrl-C to stop)")
time.sleep(0.4)

first = True
while True:
    gate_events, truncation_events = read_events(path)

    if not gate_events:
        print(f"\rNo gate events yet in {os.path.basename(path)} ...", end="", flush=True)
    else:
        if not first:
            plt.clear_terminal()
        draw(path, gate_events, truncation_events)
        first = False

    if not args.logfile:
        path = resolve_path()

    time.sleep(args.interval)
