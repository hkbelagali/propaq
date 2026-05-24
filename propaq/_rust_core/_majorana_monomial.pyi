class MajoranaMonomial:
    n_modes: int
    is_number_preserving: bool

    def __init__(self, modes: int, n_modes: int, is_number_preserving: bool = True) -> None: ...

    @property
    def modes(self) -> int:
        """
        Return the Majorana monomial in its binary representation, a 2n-length bitmask
        where the 2j-th bit indicates even modes, and the (2j+1)-th bit indicates odd modes.'

        Returns: 
            The Majorana monomial in its binary representation.
        """
        ...

    @property
    def length(self) -> int:
        """
        Return the number of active modes in the monomial, i.e. the number of bits set
        in the modes bitmask.

        ## NOTE: This is not the Pauli weight of the monomial, see the weight property for that.

        Returns:
            The number of active modes in the monomial.
        """
        ...

    @property
    def weight(self) -> int:
        """
        Return the Pauli weight of the monomial, i.e. the number of qubits on which the
        monomial acts nontrivially in its Pauli representation.

        This is done as follows: 

        1. First, we calculate the occupancy of even and odd modes separately. Even modes contribute a string of Z's
        followed by a single X on qubit j, while odd modes contribute a string of Z's followed by a single Y on qubit j.

        2. We then determine which qubits are touched by the monomial by taking the union of even and odd mode occupancies. 
        We check which qubits have only a single mode occupied - these are the ones that contribute a Z-string, 
        since this would cancel out if both modes were occupied. 

        3. The singly-occupied qubits contribute a string of Z's so we create the bitmask single that indicates 
        which qubits are singly occupied. If single[j] is 1, then we have Z's on qubits 0 through j-1. To see if 
        the Z-string survives, we need to check the parity of all set bits to the right of j. if the parity is odd, 
        we have a Z-string. We can use a prefix XOR to compute the parity efficiently: https://www.geeksforgeeks.org/dsa/prefix-xor-array/

        4. Add the single-qubit contributions of doubly-occupied qubits (1 site) and singly-occupied qubits (potentially a Z-string) to get the total weight.

        Returns:
            The Pauli weight of the monomial.
        """
        ...

    def overlap(self, other: MajoranaMonomial) -> int: 
        """
        Computes the overlap between two Majorana monomials, defined as the number of modes that they both touch.

        Arguments: 
            other: MajoranaMonomial to compute the overlap with

        Returns:
            The number of modes that both monomials touch.
        """
        ...

    def commutes_with(self, other: MajoranaMonomial) -> bool: 
        """
        Determine if two Majorana monomials commute.'

        We know that for two monomials A and B, 

        AB = (-1)^(length_A * length_B + overlap(A, B)) BA

        Therefore, if the exponent is even, the monomials commute. Since we've already 
        implemented length and overlap, we can just check the parity of the sum to determine 
        if the monomials commute.

        Arguments:
            other: MajoranaMonomial to check commutation with

        Returns:
            True if the monomials commute, False otherwise.
        """
        ...

    def resulting_weight(self, other: MajoranaMonomial) -> int:
        """
        Compute the weight of the resulting Majorana monomial obtained by multiplying two Majorana monomials.

        Arguments: 
            other: MajoranaMonomial to multiply with

        Returns:
            The weight of the resulting Majorana monomial.
        """
        ...

    def __matmul__(self, other: MajoranaMonomial) -> tuple[complex, MajoranaMonomial]: 
        """
        Multiply two Majorana monomials together, returning the resulting monomial and 
        phase factor separately.

        The phase and resulting monomial are deliberately separated to ensure hashing 
        and equality checks are based solely on the monomial, and not the phase factor. 
        This becomes important for simplifying linear combinations of Majorana monomials, 
        where we would ideally like to combine terms with the same resulting monomial 
        by summing their phases.

        Arguments:
            other: MajoranaMonomial to multiply with
        
        Returns:
            A tuple containing the phase factor (a complex number) and the resulting Majorana monomial
        """
        ...
    def trace_with_fock_state(self, fock_state: int) -> float: 
        """
        Compute the trace of the Majorana monomial with respect to a Fock state.
        This is used in computing the expectation value of the resulting 
        linear combination of Majorana monomials after circuit evolution.

        We want to compute tr(<n|M|n>) where |n> is the Fock state, and M is the Majorana monomial.
        If the monomial contains any unpaired modes, then the trace is zero because this 
        produces a Pauli string with X or Y components.

        What we're left with is a bunch of paired modes, which are products of number operators. 
        Therefore, we get 
            <n|M|n> = i^|S| Prod_{k in S} (2*n_k - 1)

        Arguments:
            fock_state: The Fock state to trace with respect to.

        Returns:
            The trace of the Majorana monomial with respect to the Fock state.
        """
        ...
    def to_bytes(self) -> bytes: 
        """
        Convert the Majorana monomial to a byte string.

        ## NOTE: This is in little-endian format.

        Returns:
            The byte string representing the Majorana monomial.
        """
        ...
    def __hash__(self) -> int: 
        """
        Compute the hash of the Majorana monomial.

        Returns:
            The hash of the Majorana monomial.
        """
        ...
    def __eq__(self, other: object) -> bool: 
        """
        Determine if two Majorana monomials are equal.

        NOTE: We check for the equality of modes modulo phase.
        
        Arguments:
            other: The other Majorana monomial to compare with.

        Returns:
            True if the monomials are equal, False otherwise.
        """
        ...
