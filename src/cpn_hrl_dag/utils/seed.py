"""Central deterministic seed setup."""
from __future__ import annotations
import os, random
import numpy as np

def seed_everything(seed:int, deterministic:bool=True, disable_cudnn:bool=False)->None:
    """Seed Python, NumPy and Torch, with an optional CuDNN-safe mode.

    ``disable_cudnn`` is intentionally opt-in: some Windows CUDA/CuDNN
    combinations have produced asynchronous illegal-address failures during
    long variable-length LSTM PPO updates.  The native PyTorch CUDA LSTM path
    is slower but avoids that backend-specific failure without changing model
    or scheduling semantics.
    """
    if seed<0: raise ValueError('seed must be non-negative')
    os.environ['PYTHONHASHSEED']=str(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark=False
        if disable_cudnn:
            torch.backends.cudnn.enabled=False
    except ImportError: pass
