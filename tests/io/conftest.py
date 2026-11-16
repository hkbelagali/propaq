"""Fixtures for the term file tests."""

import pytest


@pytest.fixture(scope="session")
def traced():
    """Compiled generic modules of the traced Pauli, Majorana and Z4 programs."""
    build = pytest.importorskip("propaq.tracer.build")
    from propaq.tracer.bases import ALL

    mods = {}
    for name in ("pauli", "majorana", "z4"):
        spec = ALL[name]()
        try:
            ir = build.compile_spec(spec)
            res = build.build_basis(spec, ir=ir, jobs=2)
        except build.BuildUnavailable as exc:
            pytest.skip(str(exc))
        mods[name] = build.import_extension(res.so_path, build.module_name(ir))
        mods[name].ir = ir
    return mods
