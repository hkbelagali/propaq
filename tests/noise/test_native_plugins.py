"""
Native plugin ABI
"""

import ctypes
import math
import shutil
import subprocess
import sys

import pytest

from propaq.circuits import PauliCircuit
from propaq.circuits.pauli.rotation import PauliRotation
from propaq.datatypes import PauliString, PauliTermSum
from propaq.datatypes._abstract import BitMask
from propaq.noise import UniformNoiseModel
from propaq.noise.native import NativeNoiseModel
from propaq.propagators.pauli import PauliPropagator
from propaq.truncation import NativeTruncator

N = 4  # qubits in every circuit below

PLUGIN_ROOT = "examples/plugins/c"
BASIS_PAULI = 0

DEPENDS_KEY = 1
DEPENDS_LAYER = 2

pytestmark = pytest.mark.skipif(
    shutil.which("cc") is None or sys.platform == "win32",
    reason="needs a C compiler to build the example plugins",
)


@pytest.fixture(scope="module")
def build_plugin(tmp_path_factory):
    """Compiles one of the C example plugins and returns the .so path."""
    out_dir = tmp_path_factory.mktemp("plugins")

    def build(relative_source: str) -> str:
        source = f"{PLUGIN_ROOT}/{relative_source}"
        out = out_dir / (relative_source.rsplit("/", 1)[-1].replace(".c", ".so"))
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-O2", "-o", str(out), source, "-lm"],
            check=True,
        )
        return str(out)

    return build


# A noise plugin that records how propaq called it
PROBE_SRC = r"""
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

/*
 * The key-aware path calls the plugin once per term, from every worker at
 * once, so these counters have to be atomic: plain `++` loses increments and
 * each counter loses a different set of them, which shows up as two tallies
 * that disagree even though they are bumped by the same call.
 */
static _Atomic uint64_t g_calls = 0;
static _Atomic uint64_t g_saw_words = 0;
static _Atomic uint32_t g_max_layer = 0;

uint32_t propaq_noise_abi_version(void) { return PROPAQ_NOISE_ABI_VERSION; }
uint32_t propaq_noise_depends(void) { return DEPENDS_VALUE; }

uint64_t probe_calls(void) { return atomic_load(&g_calls); }
uint64_t probe_saw_words(void) { return atomic_load(&g_saw_words); }
uint32_t probe_max_layer(void) { return atomic_load(&g_max_layer); }

double propaq_noise_factor(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)ctx; (void)basis_kind; (void)n_words; (void)n_units; (void)n_layers;
    atomic_fetch_add(&g_calls, 1);
    if (words != NULL) atomic_fetch_add(&g_saw_words, 1);
    uint32_t seen = atomic_load(&g_max_layer);
    while (layer_index > seen && !atomic_compare_exchange_weak(&g_max_layer, &seen, layer_index)) {
    }
    return exp(-0.01 * (double)weight);
}
"""


@pytest.fixture(scope="module")
def build_probe(tmp_path_factory):
    """Builds the recording plugin above with a chosen dependency mask."""
    out_dir = tmp_path_factory.mktemp("probes")

    def build(depends: int) -> tuple[str, ctypes.CDLL]:
        src = out_dir / f"probe_{depends}.c"
        src.write_text(PROBE_SRC.replace("DEPENDS_VALUE", f"{depends}u"))
        out = out_dir / f"probe_{depends}.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-O2", "-o", str(out), str(src), "-lm"],
            check=True,
        )

        lib = ctypes.CDLL(str(out))
        lib.probe_calls.restype = ctypes.c_uint64
        lib.probe_saw_words.restype = ctypes.c_uint64
        lib.probe_max_layer.restype = ctypes.c_uint32
        return str(out), lib

    return build


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def observable() -> PauliTermSum:
    return PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0110): 0.5, ps(0, 0b1000): 0.25})


def commuting_circuit() -> PauliCircuit:
    return PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])


def layered_circuit(n_layers: int) -> PauliCircuit:
    """`n_layers` non-commuting rotations, so the engine keeps them in separate layers."""
    return PauliCircuit([PauliRotation(ps(0b0001 << (i % N), 0), 0.3) for i in range(n_layers)])


def coefficients(term_sum) -> dict[tuple[int, int], float]:
    return {(int(t.x), int(t.z)): c for t, c in term_sum.items()}


def test_a_plugin_without_a_depends_symbol_declares_nothing(build_plugin):
    model = NativeNoiseModel(build_plugin("noise/uniform_noise.c"), config='{"damping": 0.1}')
    assert model.abi_version == 1
    assert model.depends == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("noise/qubit_local_noise.c", DEPENDS_KEY),
        ("noise/depth_dependent_noise.c", DEPENDS_LAYER),
    ],
)
def test_a_plugin_reports_the_mask_it_declared(build_plugin, source, expected):
    assert NativeNoiseModel(build_plugin(source)).depends == expected


def test_truncator_declarations_are_reported_the_same_way(build_plugin):
    assert NativeTruncator(build_plugin("truncation/weight_truncator.c")).depends == 0
    assert NativeTruncator(build_plugin("truncation/support_truncator.c")).depends == DEPENDS_KEY


def test_an_unknown_dependency_bit_is_refused(tmp_path):
    src = tmp_path / "future.c"
    src.write_text(PROBE_SRC.replace("DEPENDS_VALUE", "0xFFu"))
    out = tmp_path / "future.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", str(out), str(src), "-lm"], check=True)
    with pytest.raises(ValueError, match="unknown dependency bits"):
        NativeNoiseModel(str(out))


def test_declaring_nothing_tabulates_once_for_the_whole_run(build_probe):
    """A weight-only model is collapsed to a table of n_units + 1 entries, once."""
    path, lib = build_probe(0)
    before = lib.probe_calls()
    PauliPropagator(noise=NativeNoiseModel(path)).propagate(observable(), layered_circuit(6))
    assert lib.probe_calls() - before == N + 1


def test_declaring_nothing_is_handed_a_null_key(build_probe):
    path, lib = build_probe(0)
    before = lib.probe_saw_words()
    PauliPropagator(noise=NativeNoiseModel(path)).propagate(observable(), layered_circuit(4))
    assert lib.probe_saw_words() - before == 0


def test_declaring_layer_retabulates_once_per_layer(build_probe):
    """Still tabulated -- but rebuilt at each layer boundary, not once per term."""
    path, lib = build_probe(DEPENDS_LAYER)
    circuit = layered_circuit(5)
    n_layers = len(circuit.layers)
    before = lib.probe_calls()
    PauliPropagator(noise=NativeNoiseModel(path)).propagate(observable(), circuit)
    assert lib.probe_calls() - before == n_layers * (N + 1)


def test_declaring_layer_sees_a_real_layer_index_but_still_no_key(build_probe):
    path, lib = build_probe(DEPENDS_LAYER)
    circuit = layered_circuit(5)
    saw_words = lib.probe_saw_words()
    PauliPropagator(noise=NativeNoiseModel(path)).propagate(observable(), circuit)
    assert lib.probe_max_layer() == len(circuit.layers) - 1
    assert lib.probe_saw_words() - saw_words == 0


def test_declaring_key_is_called_per_term_and_handed_the_words(build_probe):
    path, lib = build_probe(DEPENDS_KEY)
    before_calls, before_words = lib.probe_calls(), lib.probe_saw_words()
    PauliPropagator(noise=NativeNoiseModel(path)).propagate(observable(), layered_circuit(4))
    calls = lib.probe_calls() - before_calls
    assert calls > N + 1, "a key-aware model is not tabulated"
    assert lib.probe_saw_words() - before_words == calls


def clifford_circuit() -> PauliCircuit:
    """A single pi/2 rotation, which is Clifford and so is normally deferred."""
    return PauliCircuit([PauliRotation(ps(0b0001, 0), math.pi / 2)])


def _terms_after_clifford(noise) -> int:
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    return len(coefficients(PauliPropagator(noise=noise).propagate(obs, clifford_circuit())))


def test_a_key_dependency_costs_clifford_deferral(build_probe):
    """
    Deferral leaves stored keys pre-conjugation, so a key-reading model forces
    the Clifford to branch instead -- which leaves the source row behind
    carrying cos(pi/2) (~6e-17).
    """
    weight_only, _ = build_probe(0)
    key_aware, _ = build_probe(DEPENDS_KEY)
    assert _terms_after_clifford(NativeNoiseModel(weight_only)) == 1
    assert _terms_after_clifford(NativeNoiseModel(key_aware)) == 2


def test_a_layer_dependency_does_not_cost_clifford_deferral(build_probe):
    """This is the point of splitting the axes: depth-dependence stays cheap."""
    layered, _ = build_probe(DEPENDS_LAYER)
    assert _terms_after_clifford(NativeNoiseModel(layered)) == 1


def test_a_weight_only_truncator_keeps_deferral(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/weight_truncator.c"), config='{"max_weight": 4}'
    )
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    evolved = coefficients(PauliPropagator(truncation=truncator).propagate(obs, clifford_circuit()))
    assert len(evolved) == 1


def test_uniform_plugin_matches_the_builtin_model_term_for_term(build_plugin):
    damping = 0.3
    plugin = NativeNoiseModel(
        build_plugin("noise/uniform_noise.c"), config=f'{{"damping": {damping}}}'
    )
    got = coefficients(PauliPropagator(noise=plugin).propagate(observable(), commuting_circuit()))
    want = coefficients(
        PauliPropagator(noise=UniformNoiseModel(damping=damping)).propagate(
            observable(), commuting_circuit()
        )
    )
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-12)


def test_qubit_local_noise_damps_only_the_masked_qubit(build_plugin):
    damping = 0.5
    plugin = NativeNoiseModel(
        build_plugin("noise/qubit_local_noise.c"),
        config=f'{{"damping": {damping}, "mask": 1}}',
    )
    evolved = coefficients(
        PauliPropagator(noise=plugin).propagate(observable(), commuting_circuit())
    )
    assert abs(evolved[(0, 0b0001)]) == pytest.approx(1.0 * math.exp(-damping), rel=1e-9)
    assert abs(evolved[(0, 0b1000)]) == pytest.approx(0.25, rel=1e-9)
    assert abs(evolved[(0, 0b0110)]) == pytest.approx(0.5, rel=1e-9)


def test_factor_term_reads_the_words(build_plugin):
    plugin = NativeNoiseModel(
        build_plugin("noise/qubit_local_noise.c"),
        config='{"damping": 0.25, "mask": 1}',
    )
    assert plugin.factor_term(BASIS_PAULI, [0b10], N, 1) == pytest.approx(math.exp(-0.25))
    assert plugin.factor_term(BASIS_PAULI, [0b1000_0000], N, 1) == pytest.approx(1.0)


def test_support_truncator_drops_the_terms_that_leave_the_region(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/support_truncator.c"),
        config='{"threshold": 0.01, "alpha": 40.0, "mask": 3}',
    )
    evolved = coefficients(
        PauliPropagator(truncation=truncator).propagate(observable(), commuting_circuit())
    )
    assert (0, 0b0001) in evolved, "Z_0 is inside the region"
    assert (0, 0b0110) not in evolved, "Z_1 Z_2 touches qubit 2"
    assert (0, 0b1000) not in evolved, "Z_3 is outside the region"


def test_keep_term_scores_support_not_weight(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/support_truncator.c"),
        config='{"threshold": 0.01, "alpha": 40.0, "mask": 3}',
    )
    assert truncator.keep_term(BASIS_PAULI, [0b10], N, 1, 1.0)
    assert not truncator.keep_term(BASIS_PAULI, [0b1000_0000], N, 1, 1.0)


def test_a_truncator_still_admits_a_child_it_scores_highly(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/support_truncator.c"),
        config='{"threshold": 1e-9, "alpha": 40.0, "mask": 3}',
    )
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0b0001, 0), math.pi / 4)])
    evolved = coefficients(PauliPropagator(truncation=truncator).propagate(obs, circuit))
    assert len(evolved) == 2, "the rotation's child is inside the region and is kept"


def _expectation(noise=None, truncation=None) -> float:
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    prop = PauliPropagator(noise=noise, truncation=truncation, n_threads=4)
    return prop.expectation_value(obs, layered_circuit(8), initial_state=0).expectation_value


def test_depth_dependent_noise_is_reproducible(build_plugin):
    path = build_plugin("noise/depth_dependent_noise.c")
    cfg = '{"damping": 0.05, "rate": 4.0}'
    values = {_expectation(noise=NativeNoiseModel(path, config=cfg)) for _ in range(5)}
    assert len(values) == 1, "the layer index is supplied, not counted, so nothing races"


def test_depth_dependent_noise_damps_more_than_its_depth_free_limit(build_plugin):
    path = build_plugin("noise/depth_dependent_noise.c")
    flat = _expectation(noise=NativeNoiseModel(path, config='{"damping": 0.05, "rate": 0.0}'))
    growing = _expectation(noise=NativeNoiseModel(path, config='{"damping": 0.05, "rate": 4.0}'))
    assert abs(growing) < abs(flat), "damping that grows with depth costs more signal"


def test_stochastic_truncator_is_reproducible(build_plugin):
    """
    Seeded from the term's key rather than a call counter, so the draw no
    longer depends on which worker reached a shared counter first.
    """
    path = build_plugin("truncation/stochastic_truncator.c")
    cfg = '{"threshold": 0.2, "seed": 7}'
    values = {_expectation(truncation=NativeTruncator(path, config=cfg)) for _ in range(5)}
    assert len(values) == 1


def test_stochastic_truncator_decides_consistently_per_key(build_plugin):
    """The same term always draws the same decision, however often it is seen."""
    truncator = NativeTruncator(
        build_plugin("truncation/stochastic_truncator.c"),
        config='{"threshold": 1.0, "seed": 7}',
    )
    for key in range(24):
        first = truncator.keep_term(BASIS_PAULI, [key], N, 1, 0.5)
        assert all(truncator.keep_term(BASIS_PAULI, [key], N, 1, 0.5) == first for _ in range(4))


def test_stochastic_truncator_still_responds_to_its_seed(build_plugin):
    """
    Asserted on the decision itself rather than on an expectation value: the
    draw is a pure function of (seed, key), and a whole-circuit result can
    easily wash out a handful of flipped terms.
    """
    path = build_plugin("truncation/stochastic_truncator.c")

    def decisions(seed: int) -> list[bool]:
        t = NativeTruncator(path, config=f'{{"threshold": 1.0, "seed": {seed}}}')
        return [t.keep_term(BASIS_PAULI, [key], N, 1, 0.5) for key in range(24)]

    assert decisions(7) != decisions(99), "a different seed draws a different sample"
    assert any(decisions(7)) and not all(decisions(7)), "coeff below threshold really is sampled"
