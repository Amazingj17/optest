"""Runtime facts persisted with every reproducible experiment."""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def runtime_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    metadata: dict[str, Any] = {'python_version': sys.version, 'os': platform.platform(), 'git_commit': commit}
    try:
        import torch
        metadata.update({'torch_version': str(torch.__version__), 'cuda_available': bool(torch.cuda.is_available()), 'cuda_version': None if torch.version.cuda is None else str(torch.version.cuda)})
    except ImportError:
        metadata['torch_version'] = None
    return metadata
