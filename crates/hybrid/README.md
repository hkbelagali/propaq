## propaq-hybrid 

This crate implements hybrid Schrödinger-Heisenberg propagation, computing the overlap of a 
quantum state $|\Psi\rangle$ with a Heisenberg-propagated observable $\sum_i c_i P_i$ as
$\mathbb{E}[\mathcal{O}] = \sum_i c_i \langle\Psi|P_i|\Psi\rangle$. **This crate is not intended for direct use by end users.** It is a dependency of the `propaq` software, and is best utilized through the Python interface.