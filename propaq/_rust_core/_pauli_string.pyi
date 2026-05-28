class PauliString:
    n_qubits: int

    def __init__(self, x: int, z: int, n_qubits: int) -> None:
        """
        Construct a Pauli monomial from X and Z bitmasks.

        Arguments:
            x: Integer bitmask where bit k is set if qubit k has an X or Y component.
            z: Integer bitmask where bit k is set if qubit k has a Z or Y component.
            n_qubits: Total number of qubits in the system.
        """
        ...

    @property
    def x(self) -> int:
        """X-component bitmask as a Python int."""
        ...

    @property
    def z(self) -> int:
        """Z-component bitmask as a Python int."""
        ...

    @property
    def weight(self) -> int:
        """Number of non-identity single-qubit Pauli operators (popcount of x | z)."""
        ...

    def commutes_with(self, other: "PauliString") -> bool:
        """
        Return True if this Pauli string commutes with *other*.

        Two Pauli strings commute iff the number of positions where they
        anticommute is even: popcount((self.x & other.z) ^ (self.z & other.x)) % 2 == 0.
        """
        ...

    def __matmul__(self, other: "PauliString") -> tuple[complex, "PauliString"]:
        """
        Multiply two Pauli strings, returning (phase, product).

        The phase factor is in {1, i, -1, -i}.  Phase and monomial are returned
        separately so that equal monomials (modulo phase) hash identically.
        """
        ...

    def trace_with_fock_state(self, fock_state: int) -> float:
        """
        Compute <fock_state|P|fock_state> for this Pauli string P.

        Returns 0.0 if P has any X or Y components (off-diagonal).
        For Z-only P, returns (-1)^popcount(z & fock_state).

        Arguments:
            fock_state: Computational basis state as a bitstring integer.
        """
        ...

    def to_bytes(self) -> bytes:
        """Serialize the monomial as little-endian X bytes concatenated with Z bytes."""
        ...

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
