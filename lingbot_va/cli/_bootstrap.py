from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_cli_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    return repo_root
