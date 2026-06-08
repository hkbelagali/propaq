"""
Benchmarks isolating noise-model and truncation overhead during propagation.

Four configs sweep from no noise/truncation to fully configured, letting you
measure how much each contributes to propagation cost.
"""

import numpy as np

from .bench_propagator import _build_majorana_circuit, _build_majorana_obs


_CONFIGS = ["none", "uniform", "truncation_only", "noise_and_truncation"]


def _make_noise_trunc(config):
    from propaq.noise import TruncationPolicy, UniformNoiseModel

    if config == "none":
        return None, None
    elif config == "uniform":
        return UniformNoiseModel(0.05), None
    elif config == "truncation_only":
        return None, TruncationPolicy(weight_cutoff=4, coeff_cutoff=1e-6)
    else:
        return UniformNoiseModel(0.05), TruncationPolicy(weight_cutoff=4, coeff_cutoff=1e-6)


class MajoranaNoiseBench:
    params = [_CONFIGS, [20], [1]]
    param_names = ["config", "n_orbs", "n_layers"]

    def setup(self, config, n_orbs, n_layers):
        rng = np.random.default_rng(42)
        self.circuit = _build_majorana_circuit(n_orbs, n_layers, rng)
        self.obs = _build_majorana_obs(n_orbs)
        self.noise, self.trunc = _make_noise_trunc(config)

    def time_expectation_value(self, config, n_orbs, n_layers):
        from propaq.propagators import MajoranaPropagator

        MajoranaPropagator(self.noise, self.trunc, n_threads=1).expectation_value(
            self.obs, self.circuit, fock_state=0
        )
