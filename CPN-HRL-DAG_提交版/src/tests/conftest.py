"""Shared pytest configuration for the DSH file sandbox.

The platform temporary directory is not writable for this workspace, so pytest
silently degrades on Windows and every ``tmp_path``-based test errors out with
``PermissionError``.  Two host quirks are worked around here, before pytest
creates its temporary base:

* the process-wide temporary root is redirected into a unique workspace
  directory, so nothing ever enumerates a directory left behind by a previous,
  differently-permissioned run; and
* pytest is stopped from creating private (``0o700``) temporary directories,
  because the sandbox denies access to those on this host.

Neither workaround changes an assertion or any production code; they only
choose where temporary files live.  ``scratch`` is available for tests that
prefer an explicit workspace directory over ``tmp_path``.
"""
from __future__ import annotations

import tempfile
import sys
import uuid
from pathlib import Path

import pytest

WORKSPACE_SCRATCH_ROOT = Path(__file__).resolve().parent.parent / '.pytest_scratch'


def _redirect_temporary_root() -> Path | None:
    session_root = WORKSPACE_SCRATCH_ROOT / f'session-{uuid.uuid4().hex[:12]}'
    try:
        session_root.mkdir(parents=True, exist_ok=True)
        probe = session_root / '.write-probe'
        probe.write_text('ok', encoding='utf-8')
    except OSError:  # pragma: no cover - fall back to the platform default
        return None
    tempfile.tempdir = str(session_root)
    return session_root


def _widen_pytest_private_directory_mode() -> None:
    """Stop pytest from creating private (``0o700``) temporary directories.

    The sandbox denies access to those on this host, which is what breaks
    ``tmp_path``.  Only directory modes change; no test content is affected.
    """
    try:
        from _pytest import tmpdir as pytest_tmpdir
        from _pytest.pathlib import make_numbered_dir

        def make_temp(factory, name, numbered):
            original_mkdir = Path.mkdir

            def permissive_mkdir(self, mode=0o777, parents=False, exist_ok=False):
                return original_mkdir(self, mode=0o777, parents=parents, exist_ok=exist_ok)

            Path.mkdir = permissive_mkdir
            try:
                if numbered:
                    return make_numbered_dir(root=factory.getbasetemp(), prefix=name, mode=0o777)
                target = factory.getbasetemp().joinpath(name)
                target.mkdir(mode=0o777)
                return target
            finally:
                Path.mkdir = original_mkdir

        pytest_tmpdir.TempPathFactory.mktemp = make_temp
    except Exception:  # pragma: no cover - defensive only
        pass


# The permission workaround is specific to the Windows development host.
# Linux/openEuler must retain pytest's normal private directory permissions.
SESSION_SCRATCH_ROOT = _redirect_temporary_root() if sys.platform == 'win32' else None
if sys.platform == 'win32':
    _widen_pytest_private_directory_mode()


@pytest.fixture()
def scratch():
    """A fresh writable directory inside the workspace for one test."""
    root = (SESSION_SCRATCH_ROOT or WORKSPACE_SCRATCH_ROOT) / f'test-{uuid.uuid4().hex[:12]}'
    root.mkdir(parents=True, exist_ok=True)
    return root
