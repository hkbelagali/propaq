"""
Benchmarks for MajoranaPropagator and PauliPropagator.
"""

import numpy as np


TRUNC_KWARGS = dict(weight_cutoff=6, coeff_cutoff=1e-6)


def _build_majorana_circuit(n_orbs, n_layers, rng):
    from propaq.circuits import MajoranaCircuit
    from propaq.circuits.majorana.rotation import MajoranaRotation
    from propaq.datatypes import MajoranaMonomial

    n_modes = 4 * n_orbs

    def orbital_rotation_layer():
        rots = []
        for i in range(n_orbs):
            for j in range(i + 1, n_orbs):
                modes = (1 << (2 * i)) | (1 << (2 * j + 1))
                gen = MajoranaMonomial(modes, n_modes=n_modes, is_number_preserving=False)
                rots.append(MajoranaRotation(gen, float(rng.uniform(-np.pi, np.pi))))
        return rots

    def diagonal_coulomb_layer():
        rots = []
        for i in range(n_orbs):
            modes = (1 << (2 * i)) | (1 << (2 * i + 1))
            gen = MajoranaMonomial(modes, n_modes=n_modes, is_number_preserving=True)
            rots.append(MajoranaRotation(gen, float(rng.uniform(-np.pi, np.pi))))
        return rots

    layers = []
    for _ in range(n_layers):
        layers.append(orbital_rotation_layer())
        layers.append(diagonal_coulomb_layer())

    return MajoranaCircuit(layers, n_modes=n_modes)


def _build_majorana_obs(n_orbs):
    from propaq.datatypes import MajoranaMonomial, MajoranaTermSum

    n_modes = 4 * n_orbs
    obs_mono = MajoranaMonomial(0b111111, n_modes=n_modes, is_number_preserving=True)
    return MajoranaTermSum({obs_mono: 1.0})


def _build_pauli_circuit(n_orbs, n_layers, rng):
    from propaq.circuits import PauliCircuit
    from propaq.circuits.pauli.rotation import PauliRotation
    from propaq.datatypes import PauliString

    n_qubits = 2 * n_orbs

    rotations = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gen = PauliString(0, 1 << q, n_qubits)
            rotations.append(PauliRotation(gen, float(rng.uniform(-np.pi, np.pi))))
        for q in range(0, n_qubits - 1, 2):
            gen = PauliString((1 << q) | (1 << (q + 1)), 0, n_qubits)
            rotations.append(PauliRotation(gen, float(rng.uniform(-np.pi, np.pi))))

    return PauliCircuit(rotations)


def _build_pauli_obs(n_orbs):
    from propaq.datatypes import PauliString, PauliTermSum

    n_qubits = 2 * n_orbs
    obs = PauliString(0, (1 << min(6, n_qubits)) - 1, n_qubits)
    return PauliTermSum({obs: 1.0})


class MajoranaPropagatorBench:
    params = [[20, 30], [1]]
    param_names = ["n_orbs", "n_layers"]

    def setup(self, n_orbs, n_layers):
        from propaq.noise import TruncationPolicy

        rng = np.random.default_rng(42)
        self.circuit = _build_majorana_circuit(n_orbs, n_layers, rng)
        self.obs = _build_majorana_obs(n_orbs)
        self.trunc = TruncationPolicy(**TRUNC_KWARGS)

    def time_propagate(self, n_orbs, n_layers):
        from propaq.propagators import MajoranaPropagator

        MajoranaPropagator(None, self.trunc, n_threads=1).propagate(self.obs, self.circuit)

    def time_expectation_value(self, n_orbs, n_layers):
        from propaq.propagators import MajoranaPropagator

        MajoranaPropagator(None, self.trunc, n_threads=1).expectation_value(
            self.obs, self.circuit, fock_state=0
        )


class PauliPropagatorBench:
    params = [[50, 80], [1]]
    param_names = ["n_orbs", "n_layers"]

    def setup(self, n_orbs, n_layers):
        from propaq.noise import TruncationPolicy

        rng = np.random.default_rng(42)
        self.circuit = _build_pauli_circuit(n_orbs, n_layers, rng)
        self.obs = _build_pauli_obs(n_orbs)
        self.trunc = TruncationPolicy(**TRUNC_KWARGS)

    def time_propagate(self, n_orbs, n_layers):
        from propaq.propagators import PauliPropagator

        PauliPropagator(None, self.trunc, n_threads=1).propagate(self.obs, self.circuit)

    def time_expectation_value(self, n_orbs, n_layers):
        from propaq.propagators import PauliPropagator

        PauliPropagator(None, self.trunc, n_threads=1).expectation_value(
            self.obs, self.circuit, fock_state=0
        )
