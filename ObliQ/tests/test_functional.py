from __future__ import annotations

import os
from pathlib import Path

import pytest

from common import PROJECT_DIR
from runtime_lib import run_from_project


def _run_project(args: list[str], tmp_path: Path) -> Path:
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(PROJECT_DIR, args + ["--outdir", str(tmp_path)])
    finally:
        os.chdir(original_cwd)
    return run_dir


@pytest.mark.parametrize(
    "config_path, solver_args",
    [
        ("configs/example_random.json", []),
        ("configs/example_graph.json", ["--solver.method", "annealing"]),
        ("configs/example_maxcut.json", ["--solver.method", "annealing"]),
    ],
)
def test_problem_generation_modes(tmp_path: Path, config_path: str, solver_args: list[str]):
    run_dir = _run_project(["--config", config_path, *solver_args], tmp_path)
    assert (run_dir / "results.json").exists()
    assert (run_dir / "solution.npy").exists()


@pytest.mark.parametrize(
    "solver_method",
    ["exhaustive", "annealing"],
)
def test_solver_variants(tmp_path: Path, solver_method: str):
    run_dir = _run_project(
        [
            "--config",
            "configs/example_random.json",
            "--solver.method",
            solver_method,
            "--solver.annealing.max-iter",
            "250",
        ],
        tmp_path,
    )
    assert (run_dir / "results.json").exists()
