# ObliQ — Photonic QUBO Circuits (SIGMETRICS 2025)

This folder reproduces the QUBO generation and baseline solving pieces of **ObliQ: Solving Quadratic Unconstrained Binary Optimization Problems on Real Photonic Quantum Machines** (SIGMETRICS 2025, Aditya Ranjan *et al.*). The upstream Perceval implementation released by the authors is vendored in `ref/` and used here to mirror their QUBO construction choices.

## What’s implemented
- QUBO builders that follow the authors’ routines for random instances (`ref/save_data.py`) and graph coloring encodings (`ref/save_data_graph/graph_to_qubo.py`).
- Classical solvers: exact exhaustive search for small sizes and a lightweight simulated-annealing baseline for larger problems.
- Shared CLI + config compatible with the repository-wide runner so experiments drop outputs in `outdir/run_*`.
- Paper context baked into the configs/README for quick recall.
- Quantum hooks: photonic runs using the authors’ Perceval circuits (`circ_init.py` for static ObliQ, `circ_vqc.py` for VQC). Select via `--solver.method obliq-static` or `--solver.method vqc` (Perceval required).

### Paper highlights (analysis)
- ObliQ encodes QUBO matrices directly into photonic beam-splitter angles (“anchor-point” static circuit) to get a one-shot candidate solution, then optionally stacks a VQC to refine it (hybrid ObliQ).
- Photon-count statistics are mapped back to binary variables via heuristic averaging and constraint-aware guessing to combat barren plateaus.
- Demonstrated on Quandela hardware (Altair/Ascella) with up to ~45% accuracy lift vs. generic VQCs, especially on graph-coloring QUBOs.
- Practical knobs: QUBO normalization before embedding, repetition of pairwise interactions to fit limited modes, and classical warm-starts from anchor-point outputs.

## How to run
```bash
cd reproduced_papers-jsenellart
python -m venv .venv
source .venv/bin/activate
pip install -r ObliQ/requirements.txt

# Random QUBO (6 variables) solved exhaustively
python implementation.py --project ObliQ --config configs/example_random.json

# Graph coloring QUBO (4 nodes, 3 colors) with annealing
python implementation.py --project ObliQ --config configs/example_graph.json --solver.method annealing

# Quantum runs — Perceval is required
python implementation.py --project ObliQ --config configs/example_random.json --solver.method obliq         # hybrid: static + VQC refinement
python implementation.py --project ObliQ --config configs/example_random.json --solver.method obliq-static # static anchor only
python implementation.py --project ObliQ --config configs/example_random.json --solver.method vqc --quantum.restarts 5
```
Random QUBOs are cached under `data/random_instances/` using the run seed (e.g., `data/random_instances/n6/seed7_low-2.0_high2.0.npy`) so different solvers compare on identical problems.

### Run on qubo-benchmark instances (no edits to the benchmark repo)
```bash
python utils/run_qubo_benchmark.py \
  --config qubo-benchmark/benchmark/config.yml \
  --solver annealing \
  --results-dir qubo-benchmark/results_obliq

# Quantum pass over the same instances
python utils/run_qubo_benchmark.py \
  --config qubo-benchmark/benchmark/config.yml \
  --quantum --quantum-method obliq-static
```
Outputs are stored under `qubo-benchmark/results_obliq/` with the same pickle schema as the benchmark’s solvers.
```

Main entry point: repository root `implementation.py` invoked with `--project ObliQ` (delegates to the shared runner). CLI schema lives in `configs/cli.json`; any flag there overrides the JSON config.

Outputs land in `outdir/run_YYYYMMDD-HHMMSS/` with:
- `config_snapshot.json` — resolved config used for the run
- `results.json` — QUBO metadata and solver metrics
- `qubo.npy` — Q matrix
- `solution.npy` — best bitstring found

## Configuration layout
- `configs/defaults.json` — shared defaults (seed, dtype, base outdir, solver knobs).
- `configs/example_random.json` — random QUBO demo.
- `configs/example_graph.json` — graph-coloring QUBO demo.
- `configs/cli.json` — CLI arguments mapped to config paths (`problem.*`, `solver.*`, etc.).

## Using the reference code
- Author drop-in lives under `ref/` (Perceval circuits, plotting, and data scripts). Our `lib/qubo.py` lifts their QUBO construction logic so you can swap in full photonic runs later without rewriting the generators. Quantum wrappers in `lib/quantum.py` call `ref/circ_init.py` (ObliQ static) and `ref/circ_vqc.py` (VQC) directly; Perceval is a required dependency.
- To explore the original circuits, start from `ref/circ_init.py` (static anchor-point embedding) and `ref/circ_vqc.py` (hybrid VQC layers).

## Next steps
1) Wire Perceval backends from `ref/` into `lib/runner.py` to compare photonic outputs against the classical solvers.  
2) Port the authors’ graph datasets and plotting scripts for side-by-side metric recreation.  
3) Extend the annealer (batch restarts, adaptive cooling) or hook into a MILP/QUBO solver for larger benchmarks.
