"""Strict YAML loading for reproducible command-line experiments."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    raw=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw,dict): raise ValueError('configuration root must be a YAML mapping')
    return raw

def config_hash(config: dict[str, Any]) -> str:
    # Runtime facts are recorded separately and must not alter the identity of
    # otherwise identical experiment inputs.
    canonical={key:value for key,value in config.items() if key!='runtime'}
    return hashlib.sha256(yaml.safe_dump(canonical,sort_keys=True).encode('utf-8')).hexdigest()
