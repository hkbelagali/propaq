# `propaq.circuits`

Circuit and rotation representations for both bases, the Qiskit and Cirq
converters, and the registry for overriding a gate's decomposition.

See the [circuits and gates guide](../guides/circuits.md).

::: propaq.circuits
    options:
      members:
        - PauliCircuit
        - MajoranaCircuit
        - SurrogatePauliCircuit
        - SurrogateMajoranaCircuit
        - PauliRotation
        - MajoranaRotation
        - SurrogateRotation
        - SurrogateMajoranaRotation
        - register_qiskit_gate
        - register_cirq_gate
        - pauli_rotation_generator
        - GateRep
        - GateDecompositionWarning
        - GateValidationError
