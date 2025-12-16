# ObliQ — Photonic QUBO Circuits (SIGMETRICS 2025)

This folder reproduces the QUBO generation and baseline solving pieces of **ObliQ: Solving Quadratic Unconstrained Binary Optimization Problems on Real Photonic Quantum Machines** (SIGMETRICS 2025, Aditya Ranjan *et al.*). The upstream Perceval implementation released by the authors is mirrored inside `lib/reference/` (trimmed modules that the runner imports directly), while the pristine vendor drop lives under `ref/` for exploratory use.

> **Reference code availability** — The authors’ full release is archived on Zenodo: <https://doi.org/10.5281/zenodo.17342911>. Download that archive and unpack it if you want the untouched sources locally; only the curated `lib/reference/` subset is tracked in this repo.

## What’s implemented
- QUBO builders that follow the authors’ routines for random instances (`lib/reference/save_data.py`) and graph coloring encodings (`lib/reference/save_data_graph.py`, ported from the original `graph_to_qubo.py`).
- MaxCut-to-QUBO conversion lifted from the CVaR-VQE benchmarking drop (`cvar-vqe/lib/qubo.py`), so you can sample Erdos-Renyi (or user-defined) graphs and plug them straight into the shared solver stack by setting `problem.type` to `maxcut`.
- Classical solvers: exact exhaustive search for small sizes and a lightweight simulated-annealing baseline for larger problems.
- Shared CLI + config compatible with the repository-wide runner so experiments drop outputs in `outdir/run_*`.
- Paper context baked into the configs/README for quick recall.
- Quantum hooks: photonic runs using the authors’ Perceval circuits (`lib/reference/circ_init.py` for static ObliQ, `lib/reference/circ_vqc.py` for VQC). Select via `--solver.method obliq-static` or `--solver.method vqc` (Perceval required).
- MerLin-enhanced VQC: `lib/quantum.py` builds a differentiable replica of the ObliQ VQC inside MerLin, giving us a fast gradient-based refinement option in addition to the original Perceval random search.

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

# MaxCut QUBO using the shared generator (Erdos-Renyi graph)
python implementation.py --project ObliQ --config configs/example_maxcut.json --solver.method annealing

# Quantum runs — Perceval is required
python implementation.py --project ObliQ --config configs/example_random.json --solver.method obliq         # hybrid: static + VQC refinement
python implementation.py --project ObliQ --config configs/example_random.json --solver.method obliq-static # static anchor only
python implementation.py --project ObliQ --config configs/example_random.json --solver.method vqc --solver.vqc.restarts 5
```
MerLin acceleration is enabled by default for `--solver.method vqc` and within hybrid ObliQ runs when `solver.configs.vqc.use_merlin` is true. Set that flag to `false` in a config file to fall back to the original Perceval-only random VQC search.
Random QUBOs are cached under `data/random_instances/` using the run seed (e.g., `data/random_instances/n6/seed7_low-2.0_high2.0.npy`) so different solvers compare on identical problems.

### Run on qubo-benchmark instances (no edits to the benchmark repo)
```bash
python utils/run_qubo_benchmark.py \
  --config qubo-benchmark/benchmark/config.yml \
  --solver.method annealing \
  --results-dir qubo-benchmark/results_obliq

# Quantum pass over the same instances
python utils/run_qubo_benchmark.py \
  --config qubo-benchmark/benchmark/config.yml \
  --solver.method obliq-static \
  --results-dir qubo-benchmark/results_obliq
```
Outputs are stored under `qubo-benchmark/results_obliq/` with the same pickle schema as the benchmark’s solvers.
Both `--config` and `--results-dir` accept absolute paths or ones relative to `ObliQ/`, so the commands above work no matter where you launch them from inside the project.
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
- `configs/example_maxcut.json` — Erdos-Renyi MaxCut demo (reuses the CVaR-VQE conversion logic).
- `configs/cli.json` — CLI arguments mapped to config paths (`problem.*`, `solver.*`, etc.).

## Using the reference code
- The active drop-in lives under `lib/reference/` (`circ_init*.py`, `circ_vqc.py`, `save_data*.py`, etc.). These modules are imported directly by `lib/quantum.py` and `lib/qubo.py`, so we stay self-contained without mutating the vendor sources.
- If you download the upstream release into `ref/` (see the Zenodo link above), you can use those Perceval scripts, experiment harnesses, and plotting utilities for comparison runs; `_import_ref_module` in `lib/quantum.py` can load additional circuit modules from that directory when needed.
- When hacking on the circuits, prefer editing the `lib/reference/` copies so that runs launched via the shared runner automatically pick up the changes while the pristine `ref/` tree remains untouched.
