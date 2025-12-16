from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import merlin
import numpy as np
import perceval as pcvl  # required dependency
import torch
from tqdm import tqdm
from runtime_lib.dtypes import dtype_torch
from .reference import circ_init as ref_circ_init
from .reference import circ_init_vqc as ref_circ_init_vqc
from .reference import circ_vqc as ref_circ_vqc

LOGGER = logging.getLogger(__name__)
REF_DIR = Path(__file__).resolve().parents[1] / "ref"


def _add_ref_to_path() -> None:
    ref_path = str(REF_DIR)
    if ref_path not in sys.path:
        sys.path.insert(0, ref_path)


def _import_ref_module(name: str) -> Any:
    _add_ref_to_path()
    return importlib.import_module(name)

@dataclass
class QuantumResult:
    solution: list[int]
    objective: float
    solver: str
    anchor_solution: list[int] | None = None
    anchor_objective: float | None = None


def _init_photo_circuit(cfg: dict, size: int, circuit_module: str, *, graph_val: int) -> Any:
    if circuit_module == "circ_init":
        PhotoCircuit = ref_circ_init.PhotoCircuit
    elif circuit_module == "circ_init_vqc":
        PhotoCircuit = ref_circ_init_vqc.PhotoCircuit
    elif circuit_module == "circ_vqc":
        PhotoCircuit = ref_circ_vqc.PhotoCircuit
    else:
        module = _import_ref_module(circuit_module)
        PhotoCircuit = getattr(module, "PhotoCircuit")
    pc = PhotoCircuit(
        size=size,
        time_steps=int(cfg.get("time_steps", 1)),
        nsamples=int(cfg.get("nsamples", 1000)),
        real_machine=1 if cfg.get("use_remote") else 0,
        backend=cfg.get("backend", "CliffordClifford2017"),
        token=cfg.get("token", ""),
        circuit_type=circuit_module,
        train_type=cfg.get("method", "quantum"),
        graph_val=graph_val,
        num_rep=int(cfg.get("num_rep", 10)),
    )
    return pc


def solve_static_obliq(Q: np.ndarray, cfg: dict, graph_val: int = 0) -> QuantumResult:
    pc = _init_photo_circuit(cfg, Q.shape[0], "circ_init", graph_val=graph_val)
    # Static circuit ignores coeffs, uses anchor-point embedding inside build_circuit
    ones = list(range(Q.shape[0]))
    zeros: list[int] = []
    value, state = pc.test_model(Q, ones, zeros, coeffs=None)
    return QuantumResult(solution=[int(v) for v in state], objective=float(value), solver="obliq-static")


def solve_obliq(Q: np.ndarray, cfg: dict, graph_val: int = 0) -> QuantumResult:
    """Hybrid ObliQ: static anchor + VQC refinement; return VQC but record anchor."""
    static_res = solve_static_obliq(Q, cfg, graph_val=graph_val)

    anchor_input_state = static_res.solution
    LOGGER.info(
        "ObliQ anchor solution (photon config): %s | photons=%d",
        anchor_input_state,
        int(np.sum(anchor_input_state)),
    )
    vqc_cfg = dict(cfg)
    vqc_cfg["input_state"] = anchor_input_state

    vqc_res = (
        solve_vqc_merlin(Q, vqc_cfg, graph_val=graph_val)
        if cfg.get("use_merlin", True)
        else solve_obliq_random(Q, anchor_input_state, vqc_cfg, graph_val=graph_val)
    )

    # Choose best objective but record both
    if vqc_res.objective < static_res.objective:
        solution = vqc_res.solution
        objective = vqc_res.objective
        solver_name = vqc_res.solver
    else:
        solution = static_res.solution
        objective = static_res.objective
        solver_name = static_res.solver

    return QuantumResult(
        solution=solution,
        objective=objective,
        solver=solver_name,
        anchor_solution=static_res.solution,
        anchor_objective=static_res.objective,
    )


def _build_merlin_layer(
    size: int,
    shots: int | None,
    dtype: torch.dtype,
    input_state: list[int] | None = None,
) -> merlin.QuantumLayer:
    """Mirror the circ_vqc PS/BS layout with symbolic parameters for MerLin autodiff."""
    circuit = pcvl.Circuit(size)
    param = 0

    for _ in range(2):
        # Phase shifters on all modes
        for i in range(size):
            circuit.add(i, pcvl.PS(pcvl.P(f"p{param}")))
            param += 1

        # Forward nearest-neighbor BS layer
        for i in range(size - 1):
            circuit.add((i, i + 1), pcvl.BS(pcvl.P(f"p{param}")))
            param += 1

        # Another PS layer
        for i in range(size):
            circuit.add(i, pcvl.PS(pcvl.P(f"p{param}")))
            param += 1

        # Reverse BS layer (i-1, i)
        for i in reversed(range(1, size - 1)):
            circuit.add((i - 1, i), pcvl.BS(pcvl.P(f"p{param}")))
            param += 1

    trainable_patterns = ["p"]
    in_state = input_state if input_state is not None else [1] * size
    return merlin.QuantumLayer(
        input_size=0,
        circuit=circuit,
        trainable_parameters=trainable_patterns,
        input_parameters=[],
        input_state=in_state,
        computation_space=merlin.ComputationSpace.FOCK,
        measurement_strategy=merlin.MeasurementStrategy.PROBABILITIES,
        dtype=dtype,
    )


def _merlin_train(Q: np.ndarray, cfg: dict, input_state: list[int] | None = None) -> tuple[np.ndarray, float]:
    size = Q.shape[0]
    steps = int(cfg.get("steps", 200))
    lr = float(cfg.get("lr", 0.05))
    global_dtype = cfg.get("dtype")
    dtype = dtype_torch(global_dtype) or torch.float32
    layer = _build_merlin_layer(size, shots=None, dtype=dtype, input_state=input_state)
    for p in layer.parameters():
        torch.nn.init.uniform_(p, -0.1, 0.1)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)

    Q_t = torch.tensor(Q, dtype=dtype)
    photons = int(np.sum(layer.input_state))
    comb = merlin.Combinadics("fock", photons, size)
    num_states = comb.compute_space_size()
    x_rows = []
    for state in comb.iter_states():
        x_rows.append([1 if occ > 0 else 0 for occ in state])
    x_mat = torch.tensor(x_rows, dtype=dtype)
    best_val = float("inf")
    best_state = np.zeros(size, dtype=int)

    progress = bool(cfg.get("progress", False))
    iterator = tqdm(range(steps), desc="MerLin VQC", leave=False) if progress else range(steps)

    for step in iterator:
        opt.zero_grad()
        probs = layer().to(dtype).flatten()
        if probs.shape[0] != num_states:
            raise ValueError(f"Expected {num_states} Fock states, got {probs.shape[0]}")
        energies = torch.sum(torch.matmul(x_mat, Q_t) * x_mat, dim=1)
        loss = torch.dot(probs, energies)
        loss.backward()
        opt.step()

        probs_np = probs.detach().cpu().numpy()
        occ_expect = torch.matmul(x_mat.T, probs).detach().cpu().numpy()
        cand = (occ_expect >= 0.5).astype(int)
        cand_val = float(cand @ Q @ cand)
        if cand_val < best_val:
            best_val = cand_val
            best_state = cand

        if step % max(1, steps // 10) == 0:
            LOGGER.info("MerLin VQC step %d/%d: loss=%.6f best=%.6f", step, steps, loss.item(), best_val)

    return best_state, best_val


def solve_vqc_merlin(Q: np.ndarray, cfg: dict, graph_val: int = 0) -> QuantumResult:
    input_state = cfg.get("input_state")
    state, value = _merlin_train(Q, cfg, input_state=input_state)
    return QuantumResult(solution=state.tolist(), objective=float(value), solver="vqc-merlin")


def solve_vqc_random(Q: np.ndarray, cfg: dict, graph_val: int = 0) -> QuantumResult:
    pc = _init_photo_circuit(cfg, Q.shape[0], "circ_vqc", graph_val=graph_val)
    param_count = 2 * (Q.shape[0] + (Q.shape[0] - 1)) * 2  # matches circ_vqc pattern
    restarts = max(1, int(cfg.get("restarts", 10)))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    input_state = cfg.get("input_state")
    if input_state is None:
        ones = list(range(Q.shape[0]))
    else:
        ones = [idx for idx, v in enumerate(input_state) if v == 1]
    zeros = [idx for idx in range(Q.shape[0]) if idx not in ones]

    best_val = float("inf")
    best_state: list[int] | None = None
    for _ in range(restarts):
        coeffs = rng.uniform(0, np.pi, size=param_count)
        value, state = pc.test_model(Q, ones, zeros, coeffs)
        if value < best_val:
            best_val = value
            best_state = [int(v) for v in state]

    assert best_state is not None
    return QuantumResult(solution=best_state, objective=float(best_val), solver="vqc-random")


def solve_obliq_random(Q: np.ndarray, anchor_state: list[int], cfg: dict, graph_val: int = 0) -> QuantumResult:
    """ObliQ integrated VQC using circ_init_vqc with anchor-guided photons and random params."""
    pc = _init_photo_circuit(cfg, Q.shape[0], "circ_init_vqc", graph_val=graph_val)
    param_count = 2 * (Q.shape[0] + (Q.shape[0] - 1)) * 2  # PS/BS sweeps
    restarts = max(1, int(cfg.get("restarts", 10)))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    ones = [idx for idx, v in enumerate(anchor_state) if v == 1]
    zeros = [idx for idx in range(Q.shape[0]) if idx not in ones]

    best_val = float("inf")
    best_state: list[int] | None = None
    for _ in range(restarts):
        coeffs = rng.uniform(0, np.pi, size=param_count)
        value, state = pc.test_model(Q, ones, zeros, coeffs)
        if value < best_val:
            best_val = value
            best_state = [int(v) for v in state]

    assert best_state is not None
    return QuantumResult(
        solution=best_state,
        objective=float(best_val),
        solver="obliq-vqc-random",
        anchor_solution=anchor_state,
        anchor_objective=None,
    )


def maybe_run_quantum(instance_matrix: np.ndarray, cfg: dict, *, constant: float, graph_val: int = 0) -> QuantumResult:
    method = cfg.get("method", "obliq-static").lower()
    if method == "obliq":
        result = solve_obliq(instance_matrix, cfg, graph_val=graph_val)
    elif method == "obliq-static":
        result = solve_static_obliq(instance_matrix, cfg, graph_val=graph_val)
    elif method == "vqc":
        if cfg.get("use_merlin", True):
            result = solve_vqc_merlin(instance_matrix, cfg, graph_val=graph_val)
        else:
            result = solve_vqc_random(instance_matrix, cfg, graph_val=graph_val)
    else:
        raise ValueError(f"Unknown quantum method: {method}")

    result.objective += float(constant)
    return result
