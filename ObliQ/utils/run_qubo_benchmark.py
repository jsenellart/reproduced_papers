from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import yaml

CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[2]
for candidate in (PROJECT_DIR, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lib.qubo import QuboInstance, solve_problem
from lib.quantum import maybe_run_quantum
from runtime_lib.cli import apply_cli_overrides, build_cli_parser
from runtime_lib.config import load_config

CLI_SCHEMA_PATH = PROJECT_DIR / "configs" / "cli.json"
DEFAULTS_PATH = PROJECT_DIR / "configs" / "defaults.json"
BENCH_DIR = PROJECT_DIR / "qubo-benchmark"
DEFAULT_BENCHMARK_CONFIG = BENCH_DIR / "benchmark" / "config.yml"
INST_DIR = BENCH_DIR / "instances"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results_obliq"
QUANTUM_METHODS = {"obliq", "obliq-static", "vqc"}


def _load_cli_schema() -> dict:
    return json.loads(CLI_SCHEMA_PATH.read_text(encoding="utf-8"))


def _build_config_from_cli(args: argparse.Namespace, arg_defs: list[dict]) -> dict:
    defaults = load_config(DEFAULTS_PATH)
    cfg = copy.deepcopy(defaults)
    cfg = apply_cli_overrides(cfg, args, arg_defs, PROJECT_DIR, Path.cwd())
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = int(args.seed)
    return cfg


def _prepare_method_cfg(solver_cfg: dict, seed: int) -> tuple[str, dict]:
    method = (solver_cfg.get("method") or "exhaustive").lower()
    method_cfg = dict(solver_cfg.get("configs", {}).get(method, {}))
    method_cfg.setdefault("seed", seed)
    method_cfg["method"] = method
    return method, method_cfg


def _resolve_config_path(raw: str | Path) -> Path:
    config_path = Path(raw)
    if config_path.is_absolute():
        return config_path

    project_candidate = (PROJECT_DIR / config_path).resolve()
    if project_candidate.exists():
        return project_candidate

    bench_candidate = (BENCH_DIR / config_path).resolve()
    if bench_candidate.exists():
        return bench_candidate

    raise FileNotFoundError(f"Benchmark config not found: {config_path}")


def _resolve_results_dir(raw: str | Path) -> Path:
    results_path = Path(raw)
    if results_path.is_absolute():
        return results_path
    return (PROJECT_DIR / results_path).resolve()


def load_problem(path: str | Path) -> np.ndarray:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        i = data["i"]
        j = data["j"]
        Jij = data["Jij"]
    size = int(max(np.max(i), np.max(j)) + 1)
    Q = np.zeros((size, size), dtype=float)
    Q[i, j] = Jij
    if np.any(Q != Q.T):
        Q = Q + Q.T
    return Q


def collect_instances(instances: Iterable[str]) -> list[Path]:
    problems: list[Path] = []
    for entry in instances:
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = INST_DIR / entry_path
        if entry_path.suffix == ".npz" and entry_path.is_file():
            problems.append(entry_path)
            continue
        if entry_path.is_dir():
            problems.extend(sorted(entry_path.rglob("*.npz")))
            continue
        raise FileNotFoundError(f"Instance path not found: {entry_path}")
    return sorted(problems)


def run_solver_on_problem(
    Q: np.ndarray,
    solver_cfg: dict,
    seed: int,
) -> tuple[np.ndarray, float, str]:
    method, method_cfg = _prepare_method_cfg(solver_cfg, seed)
    instance = QuboInstance(matrix=Q, constant=0.0, description="benchmark")

    if method in QUANTUM_METHODS:
        q_res = maybe_run_quantum(
            instance.matrix,
            method_cfg,
            constant=instance.constant,
            graph_val=int(method_cfg.get("graph_val", 0)),
        )
        return np.asarray(q_res.solution, dtype=int), float(q_res.objective), f"obliq_{q_res.solver}"

    solution, value, solved_method = solve_problem(instance, method_cfg, seed=seed)
    return np.asarray(solution, dtype=int), float(value), f"obliq_{solved_method}"


def run_benchmark(
    instances: list[str],
    solver_cfg: dict,
    results_dir: Path,
    solver_label: str,
    seed: int,
) -> None:
    problems = collect_instances(instances)
    if not problems:
        raise ValueError("No benchmark instances found")

    for prob in problems:
        Q = load_problem(prob)
        start = time.time()
        solution, loss, method = run_solver_on_problem(
            Q,
            solver_cfg,
            seed,
        )
        elapsed = time.time() - start

        try:
            rel = prob.relative_to(INST_DIR).with_suffix("")
        except ValueError:
            rel = prob.with_suffix("")
        res_path = results_dir / rel.parent
        res_path.mkdir(parents=True, exist_ok=True)
        res_file = res_path / f"{rel.name}_{solver_label}.pkl"

        G = nx.from_numpy_array(Q)
        payload = {
            "task": rel.name,
            "solver": solver_label,
            "loss": float(loss),
            "time": elapsed,
            "x": solution,
            "success": bool(np.all(solution >= 0)),
            "bipartite": nx.algorithms.bipartite.is_bipartite(G),
            "planar": nx.check_planarity(G)[0],
            "method": method,
        }

        with open(res_file, "wb") as fh:
            pickle.dump(payload, fh)

        print(f"[obliq] {rel.name} -> {res_file} (loss={loss:.4f}, time={elapsed:.2f}s)")


def parse_args() -> tuple[argparse.Namespace, list[dict]]:
    cli_schema = _load_cli_schema()
    parser, arg_defs = build_cli_parser(cli_schema)
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_BENCHMARK_CONFIG),
        help="Path to benchmark config.yml",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for pickled benchmark results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed used for solver restarts",
    )
    parser.add_argument(
        "--merlin",
        dest="enable_merlin",
        action="store_true",
        help="Enable MerLin-based VQC acceleration",
    )
    parser.add_argument(
        "--no-merlin",
        dest="enable_merlin",
        action="store_false",
        help="Disable MerLin-based VQC acceleration",
    )
    parser.set_defaults(enable_merlin=True)
    parser.add_argument(
        "--photonic-restarts",
        type=int,
        default=None,
        help="Override photonic solver restarts for all quantum methods",
    )
    args = parser.parse_args()
    return args, arg_defs


def main() -> None:
    args, arg_defs = parse_args()
    benchmark_cfg_path = _resolve_config_path(args.config)

    with benchmark_cfg_path.open("r", encoding="utf-8") as fh:
        benchmark_cfg = yaml.safe_load(fh) or {}

    project_cfg = _build_config_from_cli(args, arg_defs)
    solver_cfg = project_cfg.get("solver", {})
    method = solver_cfg.get("method", "exhaustive")
    solver_label = f"obliq_{method}"

    if args.photonic_restarts is not None:
        for key in ("obliq-static", "obliq", "vqc"):
            cfg = solver_cfg.get("configs", {}).get(key)
            if isinstance(cfg, dict):
                cfg["restarts"] = args.photonic_restarts

    for key in ("obliq", "vqc"):
        cfg = solver_cfg.get("configs", {}).get(key)
        if isinstance(cfg, dict):
            cfg["use_merlin"] = bool(args.enable_merlin)

    seed = int(project_cfg.get("seed", args.seed or 0))
    results_dir = _resolve_results_dir(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    instances = benchmark_cfg.get("instances", [])
    if not isinstance(instances, list):
        raise ValueError("Benchmark config must define an 'instances' list")

    run_benchmark(
        instances=instances,
        solver_cfg=solver_cfg,
        results_dir=results_dir,
        solver_label=solver_label,
        seed=seed,
    )


if __name__ == "__main__":
    main()
