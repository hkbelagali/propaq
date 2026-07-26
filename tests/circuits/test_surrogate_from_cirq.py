"""Tests for SurrogatePauliCircuit.from_cirq / SurrogateMajoranaCircuit.from_cirq.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")
import sympy  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from propaq import (  # noqa: E402
    MajoranaPropagator,
    MajoranaSurrogatePropagator,
    MajoranaTermSum,
    PauliPropagator,
    PauliSurrogatePropagator,
    PauliTermSum,
    SurrogateMajoranaCircuit,
    SurrogatePauliCircuit,
    VariationalSurrogateModel,
)
from propaq.circuits.majorana.circuit import MajoranaCircuit  # noqa: E402
from propaq.circuits.pauli.circuit import PauliCircuit  # noqa: E402

N_QUBITS = 3
N_MODES = 2 * N_QUBITS
OBS = SparsePauliOp("ZZI")


def _resolve(circuit: "cirq.Circuit", binding: dict) -> "cirq.Circuit":
    return cirq.resolve_parameters(circuit, cirq.ParamResolver(binding))


def _pauli_variational_model(circuit: "cirq.Circuit") -> tuple[VariationalSurrogateModel, SurrogatePauliCircuit]:
    obs = PauliTermSum.from_sparse_pauli_op(OBS)
    sc = SurrogatePauliCircuit.from_cirq(circuit)
    model = PauliSurrogatePropagator().build(obs, sc, initial_state=0)
    return VariationalSurrogateModel(model, sc.parameter_sources, sc.qiskit_parameters), sc


def _pauli_concrete_ev(circuit: "cirq.Circuit", binding: dict) -> float:
    obs = PauliTermSum.from_sparse_pauli_op(OBS)
    bound = _resolve(circuit, binding)
    pc = PauliCircuit.from_cirq(bound)
    return PauliPropagator().expectation_value(obs, pc, initial_state=0).expectation_value


def _majorana_variational_model(
    circuit: "cirq.Circuit",
) -> tuple[VariationalSurrogateModel, SurrogateMajoranaCircuit]:
    obs = MajoranaTermSum.from_sparse_pauli_op(OBS)
    sc = SurrogateMajoranaCircuit.from_cirq(circuit, n_modes=N_MODES)
    model = MajoranaSurrogatePropagator().build(obs, sc, initial_state=0)
    return VariationalSurrogateModel(model, sc.parameter_sources, sc.qiskit_parameters), sc


def _majorana_concrete_ev(circuit: "cirq.Circuit", binding: dict) -> float:
    obs = MajoranaTermSum.from_sparse_pauli_op(OBS)
    bound = _resolve(circuit, binding)
    mc = MajoranaCircuit.from_cirq(bound, n_modes=N_MODES)
    return MajoranaPropagator().expectation_value(obs, mc, initial_state=0).expectation_value


def _variational_model(kind: str, circuit: "cirq.Circuit"):
    return _pauli_variational_model(circuit) if kind == "pauli" else _majorana_variational_model(circuit)


def _concrete_ev(kind: str, circuit: "cirq.Circuit", binding: dict) -> float:
    return _pauli_concrete_ev(circuit, binding) if kind == "pauli" else _majorana_concrete_ev(circuit, binding)


def _from_cirq(kind: str, circuit: "cirq.Circuit"):
    if kind == "pauli":
        return SurrogatePauliCircuit.from_cirq(circuit)
    return SurrogateMajoranaCircuit.from_cirq(circuit, n_modes=N_MODES)


def _bare_parameter_circuit() -> tuple["cirq.Circuit", "sympy.Symbol"]:
    theta = sympy.Symbol("theta")
    q = cirq.LineQubit.range(N_QUBITS)
    # rz(theta) used on two different qubits: same symbol, same scale (rz's
    # angle formula is angle=theta directly, no scaling) -> shares a param_index.
    circuit = cirq.Circuit([
        cirq.rz(theta)(q[0]),
        cirq.rz(theta)(q[1]),
        cirq.rz(0.4)(q[2]),  # concrete float mixed in -> its own constant slot
    ])
    return circuit, theta


def _xx_plus_yy_circuit() -> tuple["cirq.Circuit", "sympy.Symbol", "sympy.Symbol"]:
    theta = sympy.Symbol("theta")
    beta = sympy.Symbol("beta")
    q = cirq.LineQubit.range(N_QUBITS)
    circuit = cirq.Circuit([cirq.PhasedISwapPowGate(phase_exponent=beta, exponent=theta)(q[0], q[1])])
    return circuit, theta, beta


def _affine_circuit() -> tuple["cirq.Circuit", "sympy.Symbol", "sympy.Symbol"]:
    theta = sympy.Symbol("theta")
    phi = sympy.Symbol("phi")
    q = cirq.LineQubit.range(N_QUBITS)
    circuit = cirq.Circuit([cirq.rz(2 * theta + phi + 1)(q[0])])
    return circuit, theta, phi


def _x_and_swap_circuit() -> tuple["cirq.Circuit", "sympy.Symbol"]:
    theta = sympy.Symbol("theta")
    q = cirq.LineQubit.range(N_QUBITS)
    circuit = cirq.Circuit([cirq.X(q[0]), cirq.SWAP(q[0], q[1]), cirq.rz(theta)(q[2])])
    return circuit, theta


def _random_arbitrary_gate_circuit(seed: int) -> tuple["cirq.Circuit", list["sympy.Symbol"]]:
    """Small circuit mixing arbitrary (non-native) gates with free symbols."""
    theta = sympy.Symbol("theta")
    phi = sympy.Symbol("phi")
    rng = np.random.default_rng(seed)
    q = cirq.LineQubit.range(N_QUBITS)
    circuit = cirq.Circuit([
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.rz(theta)(q[1]),
        cirq.T(q[2]),
        cirq.PhasedISwapPowGate(
            phase_exponent=phi, exponent=float(rng.uniform(0, 2 * np.pi))
        )(q[0], q[2]),
        cirq.ry(theta + phi)(q[0]),
    ])
    return circuit, [theta, phi]


@pytest.mark.parametrize("kind", ["pauli", "majorana"])
class TestFromCirqAgreesWithConcrete:
    def test_bare_parameter(self, kind):
        circuit, theta = _bare_parameter_circuit()
        variational, _ = _variational_model(kind, circuit)
        rng = np.random.default_rng(0)
        for _ in range(5):
            v = float(rng.uniform(-np.pi, np.pi))
            got = variational.evaluate({theta: v})
            want = _concrete_ev(kind, circuit, {theta: v})
            assert got == pytest.approx(want, abs=1e-9)

    def test_shared_parameter_collapses_n_params(self, kind):
        circuit, theta = _bare_parameter_circuit()
        variational, sc = _variational_model(kind, circuit)
        assert sc.n_params == 1
        assert variational.n_params == 1
        assert variational.parameters == (theta,)

    def test_numeric_gates_stay_numeric(self, kind):
        theta = sympy.Symbol("theta")
        q = cirq.LineQubit.range(N_QUBITS)
        circuit = cirq.Circuit([
            cirq.rz(0.7)(q[0]),               # pure numeric
            cirq.rx(1.3)(q[1]),                # pure numeric
            cirq.rz(2 * theta + 0.5)(q[2]),   # symbolic slot for theta + numeric constant 0.5
        ])
        _, sc = _variational_model(kind, circuit)

        rotations = sc.rotations
        numeric = [r for r in rotations if r.param_index is None]
        symbolic = [r for r in rotations if r.param_index is not None]

        assert numeric, "expected numeric rotations for the concrete-angle gates"
        assert all(r.angle is not None for r in numeric)
        assert sc.n_params == 1
        assert all(r.angle is None for r in symbolic)
        assert all(src.parameter is not None for src in sc.parameter_sources)

    def test_xx_plus_yy_symbolic_beta(self, kind):
        circuit, theta, beta = _xx_plus_yy_circuit()
        variational, _ = _variational_model(kind, circuit)
        rng = np.random.default_rng(1)
        for _ in range(5):
            binding = {
                theta: float(rng.uniform(-np.pi, np.pi)),
                beta: float(rng.uniform(-np.pi, np.pi)),
            }
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, circuit, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_affine_multi_parameter_expression(self, kind):
        circuit, theta, phi = _affine_circuit()
        variational, _ = _variational_model(kind, circuit)
        rng = np.random.default_rng(2)
        for _ in range(5):
            binding = {
                theta: float(rng.uniform(-np.pi, np.pi)),
                phi: float(rng.uniform(-np.pi, np.pi)),
            }
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, circuit, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_x_and_swap_gates(self, kind):
        circuit, theta = _x_and_swap_circuit()
        variational, _ = _variational_model(kind, circuit)
        rng = np.random.default_rng(3)
        for _ in range(5):
            binding = {theta: float(rng.uniform(-np.pi, np.pi))}
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, circuit, binding)
            assert got == pytest.approx(want, abs=1e-9)

    def test_positional_evaluate_matches_dict(self, kind):
        circuit, theta, phi = _affine_circuit()
        variational, _ = _variational_model(kind, circuit)
        binding = {theta: 0.7, phi: -0.3}
        by_dict = variational.evaluate(binding)
        by_seq = variational.evaluate([binding[p] for p in variational.parameters])
        assert by_seq == pytest.approx(by_dict)

    def test_non_affine_expression_raises(self, kind):
        theta = sympy.Symbol("theta")
        q = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.rz(theta * theta)(q[0])])
        with pytest.raises(ValueError, match="not affine"):
            _from_cirq(kind, circuit)

    def test_non_unitary_op_raises(self, kind):
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.ResetChannel()(q)])
        with pytest.raises(ValueError, match="non-unitary"):
            _from_cirq(kind, circuit)

    def test_previously_unsupported_gate_now_decomposes(self, kind):
        from propaq.circuits._cirq_gates import _decompose_cache
        _decompose_cache.clear()

        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        with pytest.warns(UserWarning, match="not natively supported"):
            sc = _from_cirq(kind, circuit)
        assert sc.rotations

    @pytest.mark.parametrize("seed", range(3))
    def test_random_arbitrary_gate_circuit_matches_concrete(self, kind, seed):
        circuit, params = _random_arbitrary_gate_circuit(seed)
        variational, _ = _variational_model(kind, circuit)
        rng = np.random.default_rng(seed + 10)
        for _ in range(5):
            binding = {p: float(rng.uniform(-np.pi, np.pi)) for p in params}
            got = variational.evaluate(binding)
            want = _concrete_ev(kind, circuit, binding)
            assert got == pytest.approx(want, abs=1e-9)
