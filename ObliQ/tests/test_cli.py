from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import build_project_cli_parser, load_runtime_ready_config
from lib import runner


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifacts(tmp_path: Path):
    cfg = load_runtime_ready_config()
    cfg["problem"]["size"] = 3
    cfg["solver"]["method"] = "exhaustive"

    runner.train_and_evaluate(cfg, tmp_path)

    results_path = tmp_path / "results.json"
    assert results_path.exists(), "Expected results.json to be created"

    contents = json.loads(results_path.read_text())
    assert "objective" in contents
    assert (tmp_path / "qubo.npy").exists()
    assert (tmp_path / "solution.npy").exists()


def test_maxcut_generation(tmp_path: Path):
    cfg = load_runtime_ready_config()
    cfg["problem"]["type"] = "maxcut"
    cfg["problem"].setdefault("graph", {})
    cfg["problem"]["graph"]["num_nodes"] = 4
    cfg["solver"]["method"] = "exhaustive"

    runner.train_and_evaluate(cfg, tmp_path)

    assert (tmp_path / "results.json").exists()
