# Property tests

Dense-matrix tests for operator bases. They need only numpy, scipy and hypothesis.

| File | What it checks |
|---|---|
| `dense.py` | Dense references for Pauli, Majorana (Jordan-Wigner) and the Z4 Weyl qudit. Also expansion, damping and Heisenberg evolution. |
| `conformance.py` | `check_basis_conformance`: checks `character`, `product`, `weight` and `trace` callables on raw words against the dense reference. |
| `test_algebra_laws.py` | Commutation, product phase, weight, trace, associativity, bicharacter law. |
| `test_end_to_end.py` | Random circuits with no truncation and no noise, against exact dense evolution. |
| `test_noise.py` | `UniformNoiseModel` against a dense damping reference. |
| `test_threads.py` | `n_threads` in {1, 2, 4} give equal term counts and values (1e-12). |

## Use for a new basis

```python
from tests.property.conformance import check_basis_conformance
check_basis_conformance("z4_qudit", 2, character=f, product=g, weight=h, trace=k)
```

Callables take raw words (tuples of Python ints). A callable may accept the
keyword `n_units`. `product` returns `(out_words, e)` with phase `exp(2 pi i e / D)`.
`trace` returns `(nonzero, e)` or a number. See `conformance.py`.
`z4_reference_callables(n)` is an integer-arithmetic Z4 reference.
`native_callables("pauli" | "majorana", n)` wraps the built-in classes.

For Majorana, `n_units` is the number of sites (qubits). The word has `2 * n_units` bits.

## Conventions, verified on v0.1.8

* Qubit q is bit q of the basis-state index. The matrix is `kron(op_{n-1}, ..., op_0)`.
* Pauli: bit `2q` is x, bit `2q+1` is z. `P = i^(x.z) X^x Z^z`.
* Majorana: `gamma_{2k} = Z..Z X_k`, `gamma_{2k+1} = Z..Z Y_k`. The monomial is
  `i^r gamma_s1 ... gamma_sm` (ascending), r = 0 for `m mod 4` in {0, 1}, else 1.
* Z4: unit bits are `x0, x1, z0, z1`. `W = X^x Z^z`, `X|j> = |j+1>`, `Z = diag(i^j)`.
  `W_a W_b = i^(z_a . x_b) W_(a+b)`. `W_g T = i^(z_g.x_t - z_t.x_g) T W_g`.
* Noise: one damping `exp(-gamma w)` per layer, applied before the layer is
  conjugated, layers walked last to first. Gates inside a layer must commute.
* Rotation: `U = exp(-i theta H / 2)` with `H = (W + W^dag) / 2`; `U^dag O U` is the Heisenberg step.

Run from the repo root with the wheel under test (the `-P` flag matters):

    <venv>/bin/python -P -m pytest tests/property -q -p no:cacheprovider
