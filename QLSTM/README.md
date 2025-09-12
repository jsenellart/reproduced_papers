# Quantum Long Short-Term Memory (QLSTM) — Reproduction

Reproduction workspace for the paper:

* Title: Quantum Long Short-Term Memory
* arXiv: https://arxiv.org/abs/2009.01783
* Goal here: clean, modular PyTorch + PennyLane reimplementation with transparent dataset generators, experiment scripts and test coverage.

## Layout

* `implementation.py` — CLI entrypoint (train LSTM or QLSTM on a chosen generator)
* `lib/model.py` — Classical LSTM cell, Quantum LSTM cell (gate VQCs), sequence wrapper
* `lib/dataset.py` — Synthetic generators + CSV loader
* `lib/rendering.py` — Minimal plotting + pickle utilities
* `data/` — External or sample CSV time‑series (e.g. `example_series.csv`)
* `requirements.txt` — Python dependencies (Torch, PennyLane, SciPy, scikit‑learn, matplotlib)
* `tests/` — (to be expanded) unit tests
* `experiments/` — Output directory automatically created per run

## Generators

Liste actuelle (clé à utiliser via `--generator`):

| Clé | Description | Paramètres principaux | Remarques |
|-----|-------------|-----------------------|-----------|
| `sin` | Sinusoïde | `frequency, amplitude, phase, n_points, noise_std` | |
| `cos` | Cosinus | idem sin | |
| `linear` | Tendance linéaire | `slope, intercept` | |
| `exp` | Croissance exponentielle | `growth_rate, initial_value` | |
| `besselj2` | Bessel J2 | `amplitude, x_scale, x_max` | Singularité évitée à x=0.1 |
| `pop_inv` | Inversion de population (cos ωt) | `omega, amplitude, t_max` | Jaynes-Cummings simple |
| `pop_inv_cr` | Inversion population (collapse & revival) | `mean_n, g, t_max, n_points` | Somme pondérée Poisson |
| `damped_shm` | Oscillateur amorti | `b, g, l, m, theta_0` | EDO intégrée |
| `logsine` | log(t+1)*sin(t) | `t_max, n_points` | Amplitude non stationnaire |
| `ma_noise` | Bruit lissé MA | `window, seed` | Autocorrélation |
| `csv` | Série externe CSV | `--csv-path <fichier>` | Prend 2e colonne si dispo |

Exemple CSV inclus : `data/airline-passengers.csv` (série passagers mensuels 1949–1960). Pour l'utiliser :

```bash
python implementation.py --model lstm --generator csv --csv-path data/airline-passengers.csv --seq-length 12 --epochs 50
```

Lister dynamiquement les générateurs disponibles :

```bash
python -c "from lib.dataset import data; print(data.list())"
```

Tous les signaux sont mis à l'échelle dans [-1, 1] (MinMaxScaler). Une fenêtre de longueur `seq_length` prédit la valeur suivante.

À envisager (selon le papier / besoins): multi-fréquences, séries chaotiques (Mackey-Glass, Lorenz), composite bruit + tendance, datasets publics supplémentaires.

## Model Overview

The Quantum LSTM replaces each classical gate (input/forget/cell/output) with a variational quantum circuit (VQC) producing hidden-size expectation values. Current configuration:

* Ansatz: Hadamard layer → feature RY embedding → (entangling + param RY)*depth
* Device: PennyLane `default.qubit` (can be swapped through `build_model` argument)
* Parameter shape: `[depth, n_qubits]` with `n_qubits = input_size + hidden_size`
* Output: PauliZ expectation values on first `hidden_size` wires → gate activations

The classical baseline uses a standard LSTM-style gating implemented with linear layers for parity in hidden dimension.

## Usage

Basic training run (single epoch smoke test):

```bash
python implementation.py --model qlstm --generator sin --epochs 5 --seq-length 8 --hidden-size 4 --vqc-depth 2
```

Classical baseline:

```bash
python implementation.py --model lstm --generator damped_shm --epochs 20 --seq-length 6 --hidden-size 8
```

Custom experiment directory + save all intermediate plots:

```bash
python implementation.py --model qlstm --generator sin --epochs 50 --save-all --exp-dir runs/qlstm_sin_depth4
```

Switch (future) to CSV time‑series once CLI flag is added (concept):

```bash
python implementation.py --model qlstm --generator csv --csv-path data/example_series.csv --epochs 30
```

Artifacts stored in `experiments/<MODEL>_TS_MODEL_<GENERATOR>/`:

* Model checkpoint (`*.pth`)
* Loss and simulation plots (`*.png`)
* Pickled final loss arrays (`*_TRAINING_LOSS.pkl`, `*_TESTING_LOSS.pkl`)
* `config.json` (exact hyperparameters)

## Roadmap

Short-term:

1. Add CLI support for CSV generator (`--csv-path`)
2. Implement multi-seed experiment script + aggregate CSV (mirroring QRKD pattern)
3. Add unit tests (shape stability, determinism with fixed seed, quantum vs classical parity)
4. Vectorize or batch QNode evaluation (reduce loop over samples)
5. Add alternative quantum ansatz / depth sweeps

Medium-term:

* Benchmark speed vs gate count
* Introduce noise models (PennyLane noisy devices) for robustness
* Compare training dynamics (gate activation distributions) vs classical

## Development Notes

Data tensors use `torch.double` to stay consistent with original quantum code (PennyLane often defaults to float64). If you mix precisions, ensure devices accept `float32` or cast explicitly.

Current known limitations:

* VQC forward pass not batched — increases runtime for large batch sizes.
* No early stopping / scheduler yet.
* CSV generator assumes single numeric column.

## Initial Scope


## Quick Commands

```bash
# Install (in repo root virtualenv already active)
pip install -r requirements.txt

# Minimal quantum run
python implementation.py --model qlstm --generator sin --epochs 3

# Classical comparison
python implementation.py --model lstm --generator sin --epochs 3
```

## Next Steps


## Attribution

Original repository inspiration: https://github.com/ycchen1989/Quantum_Long_Short_Term_Memory

Adapted source retains MIT License compatibility. Each ported file includes an attribution header.
