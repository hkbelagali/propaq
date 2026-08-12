"""
MkDocs build hooks for propaq.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("mkdocs.hooks.propaq")

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Source notebook directory -> destination directory, relative to ``docs/``.
NOTEBOOK_SOURCES: dict[str, str] = {
    "examples/usage": "examples/usage",
    "examples/plugins/notebooks": "examples/plugins",
}

STRIPPED_METADATA = ("widgets", "vscode")


_GRIFFE_NOISE = (
    "No type or annotation for",
    "does not appear in the function signature",
)


class _GriffeNoiseFilter(logging.Filter):
    """Drop griffe's docstring-completeness warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for records matching a known-noisy griffe warning."""
        message = record.getMessage()
        return not any(fragment in message for fragment in _GRIFFE_NOISE)


#: Loggers griffe's docstring-completeness warnings can surface through.
_GRIFFE_LOGGERS = ("griffe", "mkdocs.plugins.griffe", "mkdocs.plugins.mkdocstrings")


def on_startup(**kwargs: Any) -> None:
    """Install the griffe log filter before mkdocstrings starts loading modules."""
    noise_filter = _GriffeNoiseFilter()
    for name in _GRIFFE_LOGGERS:
        logging.getLogger(name).addFilter(noise_filter)


def _staged_bytes(src: Path) -> bytes:
    """The exact content *src* should have once staged, metadata stripped."""
    try:
        nb: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not parse notebook %s (%s); copying verbatim", src, exc)
        return src.read_bytes()

    metadata = nb.get("metadata")
    if isinstance(metadata, dict):
        for key in STRIPPED_METADATA:
            metadata.pop(key, None)

    return (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _stage_notebook(src: Path, dest: Path) -> bool:
    """Stage *src* at *dest*, returning True only if *dest* actually changed."""
    payload = _staged_bytes(src)
    if dest.exists() and dest.read_bytes() == payload:
        return False
    dest.write_bytes(payload)
    return True


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Stage example notebooks into ``docs/`` before MkDocs collects files.

    Runs before file collection, so the staged notebooks are picked up by
    ``mkdocs-jupyter`` and resolve against the ``nav`` entries in ``mkdocs.yml``.
    """
    docs_dir = Path(config["docs_dir"])
    staged = 0
    written = 0

    for src_rel, dest_rel in NOTEBOOK_SOURCES.items():
        src_dir = REPO_ROOT / src_rel
        dest_dir = docs_dir / dest_rel

        if not src_dir.is_dir():
            log.warning("notebook source %s does not exist; skipping", src_dir)
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        notebooks = sorted(src_dir.glob("*.ipynb"))

        expected = {notebook.name for notebook in notebooks}
        for stale in dest_dir.glob("*.ipynb"):
            if stale.name not in expected:
                stale.unlink()
                written += 1

        for notebook in notebooks:
            written += _stage_notebook(notebook, dest_dir / notebook.name)
            staged += 1

    if written:
        log.info("staged %d example notebook(s) into %s (%d changed)", staged, docs_dir, written)
    else:
        log.debug("staged %d example notebook(s) into %s (unchanged)", staged, docs_dir)
