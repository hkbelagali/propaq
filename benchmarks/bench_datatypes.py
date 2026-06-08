"""
Microbenchmarks for pure-algebra operations on PauliString, MajoranaMonomial,
PauliTermSum, and MajoranaTermSum.

These are cheap (microseconds per call) — no circuits or propagation involved.
"""


def _make_pauli(x_mask, z_mask, n_qubits):
    from propaq.datatypes import PauliString
    return PauliString(x_mask, z_mask, n_qubits)


def _make_mon(modes, n_modes, is_number_preserving=True):
    from propaq.datatypes import MajoranaMonomial
    return MajoranaMonomial(modes, n_modes, is_number_preserving=is_number_preserving)


class PauliStringBench:
    params = [[4, 20, 40]]
    param_names = ["n_qubits"]

    def setup(self, n_qubits):
        lower = (1 << (n_qubits // 2)) - 1
        upper = ((1 << n_qubits) - 1) ^ lower
        self.ps1 = _make_pauli(lower, 0, n_qubits)
        self.ps2 = _make_pauli(0, upper, n_qubits)
        # A third string for a three-way matmul chain
        self.ps3 = _make_pauli(lower, upper, n_qubits)

    def time_commutes_with(self, n_qubits):
        self.ps1.commutes_with(self.ps2)

    def time_matmul(self, n_qubits):
        _ = self.ps1 @ self.ps2

    def time_matmul_chain(self, n_qubits):
        phase, mid = self.ps1 @ self.ps2
        _ = mid @ self.ps3


class MajoranaMonomialBench:
    params = [[8, 40, 80]]
    param_names = ["n_modes"]

    def setup(self, n_modes):
        # Anticommuting pair: overlap = 1 bit → anticommutes
        self.m1 = _make_mon(0b0011, n_modes)
        self.m2 = _make_mon(0b0110, n_modes)
        self.m3 = _make_mon(0b1100, n_modes)

    def time_commutes_with(self, n_modes):
        self.m1.commutes_with(self.m2)

    def time_matmul(self, n_modes):
        _ = self.m1 @ self.m2

    def time_matmul_chain(self, n_modes):
        phase, mid = self.m1 @ self.m2
        _ = mid @ self.m3


class PauliTermSumBench:
    params = [[10, 100, 1000]]
    param_names = ["n_terms"]

    def setup(self, n_terms):
        from propaq.datatypes import PauliString, PauliTermSum
        from propaq.noise import TruncationPolicy

        # n_qubits must be >= n_terms so that 1 << i lands on a unique bit for
        # each i; cycling with modulo would collapse n_terms=1000 to only ~22
        # distinct keys.
        n_qubits = n_terms + 1
        ts1 = PauliTermSum()
        ts2 = PauliTermSum()
        for i in range(n_terms):
            ps_a = PauliString(1 << i, 0, n_qubits)        # X on qubit i
            ps_b = PauliString(0, 1 << i, n_qubits)        # Z on qubit i
            ts1.add(ps_a, 1.0 / (i + 1))
            ts2.add(ps_b, 1.0 / (i + 1))

        self.ts1 = ts1
        self.ts2 = ts2
        self.extra_term = PauliString(1, 0, n_qubits)   # X on qubit 0, already in ts1
        self.trunc = TruncationPolicy(weight_cutoff=3, coeff_cutoff=0.0)

    def time_add(self, n_terms):
        self.ts1.add(self.extra_term, 1.0)

    def time_merge(self, n_terms):
        self.ts1.copy().merge(self.ts2)

    def time_norm_squared(self, n_terms):
        self.ts1.norm_squared()

    def time_truncate(self, n_terms):
        self.ts1.copy().truncate(self.trunc)


class MajoranaTermSumBench:
    params = [[10, 100, 1000]]
    param_names = ["n_terms"]

    def setup(self, n_terms):
        from propaq.datatypes import MajoranaMonomial, MajoranaTermSum
        from propaq.noise import TruncationPolicy

        n_modes = max(n_terms * 2 + 4, 16)
        ts1 = MajoranaTermSum()
        ts2 = MajoranaTermSum()
        for i in range(n_terms):
            # number-preserving monomials: pairs (2i, 2i+1)
            m_a = MajoranaMonomial((1 << (2 * i)) | (1 << (2 * i + 1)), n_modes, is_number_preserving=True)
            # shift by one pair for ts2 to get distinct keys
            j = (i + n_terms) % (n_modes // 2)
            m_b = MajoranaMonomial((1 << (2 * j)) | (1 << (2 * j + 1)), n_modes, is_number_preserving=True)
            ts1.add(m_a, 1.0 / (i + 1))
            ts2.add(m_b, 1.0 / (i + 1))

        self.ts1 = ts1
        self.ts2 = ts2
        self.extra_term = MajoranaMonomial(0b11, n_modes, is_number_preserving=True)
        self.trunc = TruncationPolicy(weight_cutoff=3, coeff_cutoff=0.0)

    def time_add(self, n_terms):
        self.ts1.add(self.extra_term, 1.0)

    def time_merge(self, n_terms):
        self.ts1.copy().merge(self.ts2)

    def time_norm_squared(self, n_terms):
        self.ts1.norm_squared()

    def time_truncate(self, n_terms):
        self.ts1.copy().truncate(self.trunc)
