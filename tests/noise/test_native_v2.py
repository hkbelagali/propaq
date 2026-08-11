"""
Native plugin ABI v2
"""

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


def ps(x: int, z: int) -> PauliString:
    return PauliString(BitMask(x), BitMask(z), N)


def observable() -> PauliTermSum:
    return PauliTermSum({ps(0, 0b0001): 1.0, ps(0, 0b0110): 0.5, ps(0, 0b1000): 0.25})


def commuting_circuit() -> PauliCircuit:
    return PauliCircuit([PauliRotation(ps(0, 0b0001), 0.4)])


def coefficients(term_sum) -> dict[tuple[int, int], float]:
    return {(int(t.x), int(t.z)): c for t, c in term_sum.items()}


def test_v2_plugin_reports_its_abi_version(build_plugin):
    model = NativeNoiseModel(build_plugin("noise/uniform_noise_v2.c"), config='{"damping": 0.1}')
    assert model.abi_version == 2


def test_v1_plugin_still_loads_and_reports_version_one(build_plugin):
    model = NativeNoiseModel(build_plugin("noise/uniform_noise.c"), config='{"damping": 0.1}')
    assert model.abi_version == 1
    assert model.damping_factor(3, 0) == pytest.approx(math.exp(-0.1 * 3))


def test_v1_and_v2_entry_points_do_not_answer_for_each_other(build_plugin):
    v1 = NativeNoiseModel(build_plugin("noise/uniform_noise.c"), config='{"damping": 0.1}')
    v2 = NativeNoiseModel(build_plugin("noise/uniform_noise_v2.c"), config='{"damping": 0.1}')
    with pytest.raises(RuntimeError):
        v1.factor_term(BASIS_PAULI, [0], N, 2)
    with pytest.raises(RuntimeError):
        v2.damping_factor(2, 0)


def test_v2_uniform_plugin_matches_the_builtin_model_term_for_term(build_plugin):
    damping = 0.3
    plugin = NativeNoiseModel(
        build_plugin("noise/uniform_noise_v2.c"), config=f'{{"damping": {damping}}}'
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


def test_qubit_local_noise_factor_term_reads_the_words(build_plugin):
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
    assert truncator.abi_version == 2
    evolved = coefficients(
        PauliPropagator(truncation=truncator).propagate(observable(), commuting_circuit())
    )
    assert (0, 0b0001) in evolved, "Z_0 is inside the region"
    assert (0, 0b0110) not in evolved, "Z_1 Z_2 touches qubit 2"
    assert (0, 0b1000) not in evolved, "Z_3 is outside the region"


def test_support_truncator_keep_term_v2_scores_support_not_weight(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/support_truncator.c"),
        config='{"threshold": 0.01, "alpha": 40.0, "mask": 3}',
    )
    assert truncator.keep_term_v2(BASIS_PAULI, [0b10], N, 1, 1.0)
    assert not truncator.keep_term_v2(BASIS_PAULI, [0b1000_0000], N, 1, 1.0)


def test_a_v2_truncator_still_admits_a_child_it_scores_highly(build_plugin):
    truncator = NativeTruncator(
        build_plugin("truncation/support_truncator.c"),
        config='{"threshold": 1e-9, "alpha": 40.0, "mask": 3}',
    )
    obs = PauliTermSum({ps(0, 0b0001): 1.0})
    circuit = PauliCircuit([PauliRotation(ps(0b0001, 0), math.pi / 4)])
    evolved = coefficients(PauliPropagator(truncation=truncator).propagate(obs, circuit))
    assert len(evolved) == 2, "the rotation's child is inside the region and is kept"
