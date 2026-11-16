"""An installed wheel builds a user basis with only cargo, from outside the repo.

Set `PROPAQ_TEST_WHEEL` to a wheel file. The test makes a fresh venv, installs
the wheel, and runs a script from an empty directory. The script builds the
Z4 basis from the vendored core (with and without surrogate classes) and round-trips a complex term file.
Without the variable the test skips.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WHEEL = os.environ.get("PROPAQ_TEST_WHEEL")

SCRIPT = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    import propaq
    from propaq.tracer import build
    from propaq.tracer.bases import ALL

    assert os.environ["EXPECT_VENV"] in propaq.__file__, propaq.__file__
    src = build.source_dir()
    vend = Path(propaq.__file__).parent / "_vendor"
    assert src == vend, (src, vend)
    assert Path.cwd().resolve() not in Path(propaq.__file__).resolve().parents
    spec = ALL["z4"]()
    ir = build.compile_spec(spec)
    res = build.build_basis(spec, ir=ir, jobs=2)
    mod = build.import_extension(res.so_path, build.module_name(ir))
    t = mod.Term([0x1], 2)
    ts = mod.TermSum()
    ts.add(t, 1 + 2j)
    ts.save("z.gz")
    assert mod.TermSum.from_file("z.gz")[t] == 1 + 2j
    # The surrogate crate is vendored too.
    res = build.build_basis(spec, ir=ir, jobs=2, surrogate=True)
    smod = build.import_extension(res.so_path, build.module_name(ir, True))
    assert hasattr(smod, "SurrogatePropagator") and hasattr(smod, "SurrogateModel")
    print("OK", build.core_path(src), build.surrogate_path(src))
    """
)


@pytest.mark.skipif(not WHEEL or shutil.which("cargo") is None, reason="needs PROPAQ_TEST_WHEEL and cargo")
def test_wheel_builds_user_basis_with_vendored_core(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    py = venv / "bin" / "python"
    subprocess.run([str(py), "-m", "pip", "install", "--no-deps", "--force-reinstall", WHEEL], check=True, capture_output=True, env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PIP_TARGET")})
    work = tmp_path / "elsewhere"
    work.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PROPAQ_SOURCE_DIR", "PYTHONPATH")}
    env["EXPECT_VENV"] = str(venv)
    env["PROPAQ_CACHE_DIR"] = str(tmp_path / "cache")
    r = subprocess.run([str(py), "-P", "-c", SCRIPT], cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "OK" in r.stdout and "_vendor" in r.stdout and "propaq-surrogate" in r.stdout
