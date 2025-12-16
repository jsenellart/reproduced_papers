from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR
from runtime_lib import run_from_project


def test_run_from_project_executes(tmp_path: Path):
    original = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                str(PROJECT_DIR / "configs" / "example_random.json"),
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        # run_from_project changes cwd to project dir; restore for pytest hygiene
        os.chdir(original)

    assert run_dir.exists()
    assert (run_dir / "results.json").exists()
