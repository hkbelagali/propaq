import matplotlib.pyplot as plt

from propaq import LogParser

filename = "benchmark.jsonl"

parser = LogParser(filename)

outbox_terms = parser.outbox_terms
map_terms = parser.map_terms

plt.figure(figsize=(6, 4))
plt.plot(map_terms, label="Local hashmaps")
plt.plot(outbox_terms, label="Outbox")
plt.xlabel("Gate index")
plt.ylabel("Term count")
plt.yscale("log")
plt.legend()
plt.savefig(f"termcount_{filename}.png")

l1_max = parser.discarded_coeff_max
l1_sum = parser.discarded_coeff_l1

plt.figure(figsize=(6, 4))
plt.plot(l1_max, "-o", ms=4, lw=1.2, label="Max discarded coeff")
plt.plot(l1_sum, "-o", ms=4, lw=1.2, label="L1 norm of discarded coeffs")
plt.xlabel("Gate index")
plt.ylabel("Coefficient magnitude")
plt.yscale("log")
plt.legend()
plt.savefig(f"discarded_coeffs_{filename}.png")