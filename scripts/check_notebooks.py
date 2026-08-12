"""
Check that the example notebooks still match propaq's public API.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories whose ``*.ipynb`` files are published in the documentation.
NOTEBOOK_DIRS = ("examples/usage", "examples/plugins/notebooks")

#: True when running inside GitHub Actions, which understands ``::error`` lines.
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@dataclass(frozen=True)
class Problem:
    """One stale reference, located at *cell* / *line* within a notebook."""

    cell: int
    line: int
    message: str


@dataclass(frozen=True)
class _Obj:
    """A resolved runtime object: a module, class or function."""

    value: Any


@dataclass(frozen=True)
class _Inst:
    """An instance of *cls*, produced by calling it or by a known return type."""

    cls: type


@dataclass(frozen=True)
class _Bound:
    """A method already bound to an instance, so its ``self`` is spoken for."""

    func: Any


def _instance_attributes(cls: type) -> frozenset[str]:
    """
    Every attribute an instance of *cls* may legitimately carry.
    """
    found: set[str] = set(dir(cls))

    for klass in getattr(cls, "__mro__", (cls,)):
        found.update(getattr(klass, "__annotations__", {}))
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):
            continue  # Rust classes and builtins have no Python source.
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "self":
                    found.add(node.attr)

    return frozenset(found)


def _code_cells(notebook: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(cell_index, source)`` for each code cell in *notebook*."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            yield index, "".join(cell.get("source", []))


def _is_propaq_owned(obj: Any) -> bool:
    """
    True when *obj* is defined by propaq.
    """
    module = getattr(obj, "__module__", None) or getattr(obj, "__name__", None)
    return isinstance(module, str) and module.split(".")[0] == "propaq"


def _signature_of(func: Any) -> inspect.Signature | None:
    """
    Signature of *func*, or None when it cannot be introspected.
    """
    try:
        return inspect.signature(func)
    except (TypeError, ValueError):
        return None


class _NotebookScope:
    """
    Tracks the propaq values bound by a notebook, cell by cell.
    """

    def __init__(self) -> None:
        self.names: dict[str, _Obj | _Inst] = {}
        self.problems: list[Problem] = []
        self._seen: set[Problem] = set()
        self._cell = 0

    # -- reporting ---------------------------------------------------------

    def _report(self, node: ast.AST, message: str) -> None:
        """
        Record a problem, ignoring a repeat of one already seen.
        """
        problem = Problem(self._cell, getattr(node, "lineno", 0), message)
        if problem not in self._seen:
            self._seen.add(problem)
            self.problems.append(problem)

    # -- resolution --------------------------------------------------------

    def _resolve(self, node: ast.expr) -> _Obj | _Inst | None:
        """Resolve *node* to a known value, or None when it cannot be pinned down."""
        if isinstance(node, ast.Name):
            return self.names.get(node.id)

        if isinstance(node, ast.Attribute):
            return self._resolve_attribute(node)

        if isinstance(node, ast.Call):
            return self._resolve_call(node)

        return None

    def _resolve_attribute(self, node: ast.Attribute) -> _Obj | _Inst | _Bound | None:
        """Resolve ``base.attr``, reporting when *attr* is missing from a propaq base."""
        base = self._resolve(node.value)
        if base is None or isinstance(base, _Bound):
            return None

        if isinstance(base, _Inst):
            cls = base.cls
            if node.attr in _instance_attributes(cls):
                value = getattr(cls, node.attr, None)
                return _Bound(value) if inspect.isfunction(value) else self._wrap(value)
            if _is_propaq_owned(cls):
                self._report(node, f"{cls.__name__}.{node.attr} does not exist")
            return None

        owner = base.value
        if hasattr(owner, node.attr):
            return self._wrap(getattr(owner, node.attr))

        if _is_propaq_owned(owner):
            label = getattr(owner, "__name__", repr(owner))
            self._report(node, f"{label}.{node.attr} does not exist")
        return None

    def _resolve_call(self, node: ast.Call) -> _Obj | _Inst | None:
        """Resolve ``f(...)``, reporting when the arguments do not fit *f*."""
        func = self._resolve(node.func)
        if func is None:
            return None

        bound = isinstance(func, _Bound)
        target = func.func if bound else func.value  # type: ignore[union-attr]
        if target is None or not callable(target):
            return None

        if _is_propaq_owned(target):
            self._check_call_signature(node, target, drop_self=bound)

        if isinstance(target, type):
            return _Inst(target)

        signature = _signature_of(target)
        if signature is not None:
            returned = signature.return_annotation
            if isinstance(returned, type):
                return _Inst(returned)
        return None

    def _check_call_signature(self, node: ast.Call, target: Any, *, drop_self: bool) -> None:
        """Report when *node*'s arguments cannot bind to *target*'s signature."""
        signature = _signature_of(target)
        if signature is None:
            return

        if drop_self:
            # Reached through an instance, so `self` is already supplied. The
            # class-level lookup handed back the plain function, which still
            # declares it.
            parameters = list(signature.parameters.values())
            if parameters and parameters[0].name in ("self", "cls"):
                signature = signature.replace(parameters=parameters[1:])

        # Star-args hide the real arity, so a call using them cannot be judged.
        if any(isinstance(a, ast.Starred) for a in node.args):
            return
        if any(k.arg is None for k in node.keywords):
            return

        # Only the *shape* of the call is checked; values are never evaluated.
        positional = [None] * len(node.args)
        keywords = {k.arg: None for k in node.keywords if k.arg is not None}

        try:
            signature.bind(*positional, **keywords)
        except TypeError as exc:
            label = getattr(target, "__qualname__", None) or getattr(target, "__name__", "?")
            self._report(node, f"{label}{signature} cannot accept this call: {exc}")

    @staticmethod
    def _wrap(value: Any) -> _Obj | None:
        """Wrap a resolved attribute, dropping values too dynamic to follow."""
        return _Obj(value) if value is not None else None

    # -- binding -----------------------------------------------------------

    def _bind(self, target: ast.expr, value: _Obj | _Inst | None) -> None:
        """
        Bind *target* to *value*, forgetting the name when value is unknown.
        """
        if not isinstance(target, ast.Name):
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    self.names.pop(sub.id, None)
            return

        if value is None:
            self.names.pop(target.id, None)
        else:
            self.names[target.id] = value

    def _bind_import_from(self, node: ast.ImportFrom) -> None:
        """Handle ``from propaq.x import A``, reporting names that no longer exist."""
        if not node.module or node.module.split(".")[0] != "propaq":
            return
        try:
            module = importlib.import_module(node.module)
        except ImportError as exc:
            self._report(node, f"cannot import module {node.module!r}: {exc}")
            return

        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            if not hasattr(module, alias.name):
                self._report(node, f"{node.module}.{alias.name} no longer exists")
                self.names.pop(bound, None)
                continue
            self.names[bound] = _Obj(getattr(module, alias.name))

    def _bind_import(self, node: ast.Import) -> None:
        """Handle ``import propaq.x [as y]``."""
        for alias in node.names:
            if alias.name.split(".")[0] != "propaq":
                continue
            try:
                module = importlib.import_module(alias.name)
            except ImportError as exc:
                self._report(node, f"cannot import module {alias.name!r}: {exc}")
                continue
            if alias.asname:
                self.names[alias.asname] = _Obj(module)
            else:
                # `import propaq.circuits` binds the root package name.
                root = alias.name.split(".")[0]
                self.names[root] = _Obj(importlib.import_module(root))

    # -- driving -----------------------------------------------------------

    def feed(self, cell: int, source: str) -> None:
        """Analyse one code cell, carrying bindings forward."""
        self._cell = cell
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self._bind_import_from(node)
            elif isinstance(node, ast.Import):
                self._bind_import(node)
            elif isinstance(node, ast.Assign):
                value = self._resolve(node.value)
                for target in node.targets:
                    self._bind(target, value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._bind(node.target, self._resolve(node.value))
            elif isinstance(node, ast.For | ast.comprehension):
                # Loop variables take one element of something unmodelled.
                self._bind(node.target, None)
            elif isinstance(node, ast.Attribute | ast.Call):
                # Reads that are not part of an assignment still get checked.
                self._resolve(node)


def _check_notebook(notebook: Path) -> list[Problem]:
    """Every stale reference in *notebook*, in cell order."""
    scope = _NotebookScope()
    for index, source in _code_cells(notebook):
        scope.feed(index, source)
    return scope.problems


def main() -> int:
    """Check every published notebook, reporting each stale reference."""
    notebooks = sorted(
        path for directory in NOTEBOOK_DIRS for path in (REPO_ROOT / directory).glob("*.ipynb")
    )

    if not notebooks:
        print("no notebooks found; nothing to check", file=sys.stderr)
        return 1

    failures = 0

    for notebook in notebooks:
        rel = notebook.relative_to(REPO_ROOT)
        for problem in _check_notebook(notebook):
            failures += 1
            message = f"{rel} (cell {problem.cell}, line {problem.line}): {problem.message}"
            if IN_GITHUB_ACTIONS:
                print(f"::error file={rel}::{message}")
            print(message, file=sys.stderr)

    if failures:
        print(
            f"\n{failures} stale reference(s) across {len(notebooks)} notebook(s).\n"
            "These notebooks are published in the documentation with their committed\n"
            "outputs, so they are showing an API that no longer exists. Update and\n"
            "re-run them, then commit the new outputs.",
            file=sys.stderr,
        )
        return 1

    print(f"checked {len(notebooks)} notebook(s): all propaq references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
