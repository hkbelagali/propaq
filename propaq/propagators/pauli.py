"""
Pauli propagators for quantum simulation.

Implements the BCH/Heisenberg-picture propagation over Pauli algebra,
analogous to the Majorana propagation described in:

[1] M. S. Rudolph, T. Jones, Y. Teng, A. Angrisani, and Z. Holmes,
    "Pauli Propagation: A Computational Framework for Simulating Quantum Systems,"
    May 27, 2025, arXiv: arXiv:2505.21606. doi: 10.48550/arXiv.2505.21606.

"""

from propaq._rust_core import PauliPropagator as PauliPropagator
