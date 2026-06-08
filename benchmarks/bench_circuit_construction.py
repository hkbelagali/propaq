"""
Benchmarks for from_qiskit circuit parsing (DAG traversal + gate decomposition).

Uses only the supported gate set: xx_plus_yy and rz.
"""


def _build_qiskit_circuit(n_qubits, n_layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    qc = QuantumCircuit(n_qubits)
    for _ in range(n_layers):
        for i in range(0, n_qubits - 1, 2):
            qc.append(XXPlusYYGate(0.3), [i, i + 1])
        for i in range(n_qubits):
            qc.rz(0.1 * i, i)

    return qc


class MajoranaCircuitFromQiskitBench:
    params = [[4, 8, 16], [1, 3]]
    param_names = ["n_qubits", "n_layers"]

    def setup(self, n_qubits, n_layers):
        self.qc = _build_qiskit_circuit(n_qubits, n_layers)
        self.n_modes = 2 * n_qubits

    def time_from_qiskit(self, n_qubits, n_layers):
        from propaq.circuits import MajoranaCircuit

        MajoranaCircuit.from_qiskit(self.qc, self.n_modes)


class PauliCircuitFromQiskitBench:
    params = [[4, 8, 16], [1, 3]]
    param_names = ["n_qubits", "n_layers"]

    def setup(self, n_qubits, n_layers):
        self.qc = _build_qiskit_circuit(n_qubits, n_layers)

    def time_from_qiskit(self, n_qubits, n_layers):
        from propaq.circuits import PauliCircuit

        PauliCircuit.from_qiskit(self.qc)
