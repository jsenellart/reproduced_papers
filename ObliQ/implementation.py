#!/usr/bin/env python3
"""Thin wrapper that delegates to the repository-wide runner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib import run_from_project


def main(argv: list[str] | None = None) -> int:
    run_from_project(PROJECT_DIR, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
