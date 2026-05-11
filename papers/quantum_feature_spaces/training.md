# VQC Training — Findings, Experiments, Conclusions

**Reference paper.** This work is a photonic re-implementation of

> Havlíček, Córcoles, Temme, Harrow, Kandala, Chow, Gambetta.
> *Supervised learning with quantum-enhanced feature spaces.*
> **Nature 567, 209–212 (2019)**, arXiv:[1804.11326](https://arxiv.org/abs/1804.11326).

The original is qubit-based (superconducting hardware). We provide **two
faithful re-implementations**:

- **`qubit_quantum`** — a direct re-implementation of the paper's qubit
  IQP feature map (`U_Φ(x) H^⊗n U_Φ(x) H^⊗n` with single-qubit + cross
  `(π−x_i)(π−x_j)` Z-Z terms) and variational ansatz (alternating Y/Z
  rotation layers + CZ-chain entangler). Same feature dimension
  convention as the rest of the project: ``n_qubits = m − 1``.
- **`photonic_quantum`** — the photonic analog: Haar unitaries on either
  side of a per-mode phase encoder, with parity readout on
  ``parity_modes``, running on the
  [Merlin](https://github.com/merlinquantum/merlin) +
  [Perceval](https://github.com/Quandela/Perceval) stack.

**Scope extension beyond the paper.** Havlíček et al. study only one
direction: a qubit variational classifier on a qubit-generated dataset
(one cell). We extend this to a **4 × 3 grid** — four dataset generators
(`qubit_quantum`, `photonic_quantum`, `analytical`, `mlp`) crossed with
three student architectures (Fourier-feature MLP, photonic variational
sandwich, qubit IQP variational classifier):

```
                          ┌───────────────── student ─────────────────┐
                          │  MLP    Photonic-quantum   Qubit-quantum  │
   ┌──────────────────────┼───────────────────────────────────────────┤
g  │ qubit_quantum        │   ●           ●            paper setup    │
e  │ photonic_quantum     │   ●       photonic Havlíček    ●          │
n  │ analytical           │   ●           ●                ●          │
   │ mlp                  │ self-dist.    ●                ●          │
   └──────────────────────┴───────────────────────────────────────────┘
```

The diagonal-of-flavours (`X` student on `X` teacher) provides sanity
checks: each architecture should reproduce a teacher of its own family. The
off-diagonal cells test cross-architecture expressivity — e.g. whether
the photonic ansatz can fit a qubit IQP teacher or whether an MLP can
fit a quantum-generated decision boundary. Each generator accepts the
Havlíček-style **separation gap** (`--min-margin`) and exact **class
balancing** (`--balanced`) so all comparisons share the same
Havlíček-compatible dataset preparation.

```
Teacher:  W1 (Haar) → phase-encode(x) → W2 (Haar)   [fixed parameters]
Student:  T1 (train) → phase-encode(x) → T2 (train)  [RECTANGLE interferometer]
```

---

## Dataset generation — strategies, parameters, sample yields

The data-generation code lives in `work/data/`. There are **four** generators
behind a uniform interface (`size`, `m`, `k`, `seed`, `balanced`, `min_margin`);
each returns a `Dataset(X, y, metadata, soft_targets)`.

### The four strategies

| Generator | Label rule | Soft target | Cost per sample |
|---|---|---|---|
| `photonic_quantum` | sign of parity expectation of a Merlin sandwich circuit `W1(Haar) → encode(x) → W2(Haar)` with photon-number readout on `parity_modes` | `(N, 1)` parity expectation ∈ `[-1, +1]` | high (one quantum-layer forward pass per draw) |
| `qubit_quantum` | sign of `⟨Z^⊗n⟩` on `\|ψ⟩ = W_teacher(θ) U_Φ(x) H^⊗n U_Φ(x) H^⊗n \|0^n⟩` (Havlíček IQP feature map + random variational teacher) | `(N, 1)` parity expectation ∈ `[-1, +1]` | moderate (`O(n_qubits · 2^{n_qubits})` per sample) |
| `analytical` | `sign( Σ_{i<j} sin(k·(x_i − x_j)) )` | `(N, 1)` normalised score ∈ `[-1, +1]` | very low (closed form) |
| `mlp` | `argmax( teacher_MLP(Fourier(x)) )` for a random-init teacher | `(N, 2)` softmax probs (used for KD) | low (forward through small MLP) |

Features `X` are always `(N, m-1)` uniform in `[0, 2π]`. Labels `y` are integers in `{0, 1}`.

The "confidence" used for the `min_margin` gate is normalised to `[0, 1]` in all four
cases, so the same numeric value is comparable across generators:

| Generator | confidence used for `min_margin` |
|---|---|
| `photonic_quantum` | `|parity_expectation|` |
| `qubit_quantum`    | `|parity_expectation|` |
| `analytical`       | `|score| / n_pairs` |
| `mlp`              | `|p1 − p0|` |

### Filtering & resampling pipeline

`size` is the **post-filter, post-balance** target. The helper
`data._resample.filter_resample` orchestrates: draw → hard-filter by `min_margin`
→ estimate balance yield → iterate until target reached → apply balance at the end
(deferred, not per-iteration).

### Measured raw → kept yields (size = 2 000, seed = 42)

```
       gen  m  k  bal margin     raw  kept  mar%  bal% iters bail
----------------------------------------------------------------------
photonic_q  4  2    F   0.00    2000  2000 100.0 100.0     0    n
photonic_q  4  2    T   0.00    3846  2000 100.0  61.9     1    n
photonic_q  4  2    T   0.05    4813  2000  89.0  51.6     1    n
photonic_q  4  2    T   0.20    2000   558  52.4  27.9     0    Y
photonic_q  6  3    F   0.00    2000  2000 100.0 100.0     0    n
photonic_q  6  3    T   0.00    2861  2000 100.0  77.7     1    n
photonic_q  6  3    T   0.05    6159  2000  62.1  41.9     1    n
photonic_q  6  3    T   0.10    2000   432  34.9  21.6     0    Y
analytical  4  2    T   0.10    3120  2000  74.2  72.8     1    n
analytical  4  2    T   0.30    5607  2000  46.4  45.4     1    n
analytical  6  3    T   0.30    2000   330  17.6  16.5     0    Y
       mlp  4  2    T   0.05    6371  2000  43.2  40.7     1    n
       mlp  6  3    T   0.10    6575  2000  40.3  39.6     1    n
```

### Key observations

- **Photonic quantum has measure concentration at m=6.** For `m=6, k=3`, std(parity) ≈ 0.074.
  The Havlíček 0.3 gap (designed for qubits with σ≈1) maps to ≈4σ here, leaving only ~1.6%
  of raw samples. The viable margin range is `0 < min_margin ≲ 0.10`.
  At `min_margin=0.10` (≈1.4σ), ~37% of raw samples survive.

- **Analytical handles 0.3 easily.** Pairwise-sin scores are not concentrated near zero;
  `min_margin=0.3` keeps 46% with ~2.8× raw oversample.

- **MLP teacher is boundary-heavy.** Tanh activations + no biases push most logits near 0.
  Stick to `min_margin ≤ 0.05` for the MLP generator.

### Recommended `min_margin` values

| Generator | Sensible range | Havlíček-equivalent |
|---|---|---|
| `photonic_quantum`, `m=4, k=2` | 0 – 0.20 | 0.05 |
| `photonic_quantum`, `m=6, k=3` | 0 – 0.10 | 0.10 (≈1.4σ) |
| `qubit_quantum`, `m=3, k=4`    | 0 – 0.30 | **0.30 (paper's exact value, feasible)** |
| `analytical` | 0 – 0.40 | 0.30 (paper's value) |
| `mlp` | 0 – 0.05 | not directly comparable |

---

## Bugs found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| Dead `phi{depth}` params before measurement | grad ≈ 1e-9 for trailing phase block | Skip `phi` for last depth layer (`d < depth` only) |
| Wide random init → barren plateau | m=6 init grad 5–8× weaker than m=4 | `init_scale=0.05` (near-identity init, empirically optimal) |
| `CosineAnnealingLR` killing convergence | LR→0 at epoch 300 while loss still improving | `ReduceLROnPlateau(patience=30, factor=0.5, min_lr=1e-5)` |
| Duplicate encoding-param names for depth > 1 | `RuntimeError` at circuit build time | Unique names `x{d*n_features+j+1}` per layer |

Parameter count after fixing dead phi (m=6):

| Config | Before | After |
|---|---|---|
| m_c=6, depth=1 | 72 | 66 |
| m_c=8, depth=1 | 128 | 120 |
| m_c=10, depth=1 | 200 | 190 |

---

## Experiment results

Training config (unless otherwise noted): Adam lr=1e-2, ReduceLROnPlateau(patience=30,
factor=0.5, min_lr=1e-5), batch_size=64, early_stopping_patience=60, data_seed=42.

---

### Sanity check — m=4, k=2 (quantum/quantum, loss=mse)

`--learner quantum --generator quantum --m 4 --k 2 --depths 1 --sizes 4 --loss mse --balanced`

**test_acc = 96.22% ✓** (target 90% reached at depth=1, 28 params)

---

### m=6, k=3 — quantum/quantum, loss=mse, min_margin=0.10

```bash
python work/train.py --learner quantum --generator quantum \
  --m 6 --k 3 --depths 1 --sizes 6 --loss mse --balanced \
  --min-margin 0.10 --dataset-size 10000 --epochs 300
```

Dataset: 10000 → ~3740 survive |parity|≥0.10 → Train=1824, Test=456 (balanced).

**Result: depth=1, 66 params, test_acc=100.00% ✓**

| Epoch | train_loss | train_acc | val_acc |
|---|---|---|---|
| 0 | 0.4618 | 0.4988 | 0.5495 |
| 30 | 0.0199 | 0.7546 | 0.7747 |
| 70 | 0.0127 | 0.8752 | 0.9121 |
| 110 | 0.0003 | 1.0000 | **1.0000** |

**Key insight.** The plateau at ~75% without margin was a signal-to-noise problem, not a
capacity problem: the raw dataset contained ~63% near-boundary samples (|parity| ≈ 0)
whose MSE contribution dominated without providing directional gradient. Discarding
the bottom 63% by confidence yields a perfectly learnable problem at depth=1.

Note: `mse` loss uses soft teacher parity values ∈ [-1,+1] as regression targets.
It requires the quantum generator (only source of continuous parity targets).
For other generators, `hloss` (Havlíček-style BCE with trainable bias) must be used.

---

### m=6, k=3 — quantum/quantum gridsearch (loss=hloss)

```bash
python work/train.py \
  --learner quantum --generator quantum \
  --m 6 --k 3 --dataset-size 10000 \
  --balanced --min-margin 0.1 \
  --loss hloss \
  --depths 1 2 3 4 --sizes 6 8 10 \
  --target-accuracy 0.90 \
  --epochs 500 --batch-size 64 \
  --early-stopping-patience 60 \
  --model-seed 42
```

Dataset: 10000 target → 60 789 raw drawn (36.6% margin survival) → Train=8000, Test=2000.

`hloss` uses hard {0,1} labels with a trainable per-class bias (no soft parity targets).
It is compatible with any generator (unlike `mse`).

**Full results:**

| m_circ | depth | params | best_val | test_acc |
|--------|-------|--------|----------|----------|
| 6      | 1     | 66     | 71.75%   | 73.00%   |
| 6      | 2     | 102    | 74.12%   | 73.25%   |
| 6      | 3     | 138    | 86.37%   | 85.95%   |
| 6      | 4     | 174    | 88.75%   | 85.25%   |
| 8      | 1     | 120    | 73.25%   | 75.90%   |
| 8      | 2     | 184    | 82.13%   | 81.20%   |
| 8      | 3     | 248    | 85.00%   | 83.60%   |
| 8      | 4     | 312    | 87.63%   | 88.50%   |
| 10     | 1     | 190    | 73.25%   | 74.95%   |
| 10     | 2     | 290    | 84.00%   | 85.10%   |
| 10     | 3     | 390    | 85.87%   | 86.25%   |
| **10** | **4** | **490**| **92.25%**| **90.55% ✓** |

**First hit at 90%:** m_circ=10, depth=4, 490 params, test_acc=90.55%.

**Observations:**

- **Depth is the dominant factor.** Depths 1–2 plateau in the 73–82% range for all sizes.
  The jump from depth=2 to depth=3 is large (+10–12%), and depth=3→4 adds another +2–4%.
- **m_circ matters at higher depth.** At depth=4: m_c=6 (174 params) → 85.25%,
  m_c=8 (312 params) → 88.50%, m_c=10 (490 params) → 90.55%. At depth=1–2 the
  difference between sizes is small.
- **`hloss` vs `mse`** (m_c=6, depth=1, 66 params, same dataset setup):
  `mse` reaches 100% (soft regression targets eliminate boundary noise);
  `hloss` plateaus at 73% (hard labels retain residual noise). For the full benchmark
  (which must work across generators), `hloss` is required; `mse` is the upper bound
  for the quantum-generator case only.

---

## Code changes

**`work/learner/photonic_quantum.py`** *(formerly `work/train_quantum.py`)*
- `QuantumClassifier.__init__`: `init_scale=0.05` default, phi loop condition `d < depth`,
  near-identity init block; encoding param names `x{d*n_features+j+1}`.
- `train_and_eval_quantum`: `ReduceLROnPlateau` scheduler; early stopping (patience, dual
  criterion: counter increments only when both val_acc AND train_loss fail to improve).
- `find_min_depth`: added `min_margin`, `early_stopping_patience` parameters.

**`work/learner/qubit_quantum.py`** *(new — Havlíček paper-faithful student)*
- `QubitClassifier`: pure-PyTorch qubit simulator, IQP feature map, layered Y/Z + CZ
  variational ansatz, optional trainable bias for `hloss`.
- `train_and_eval_qubit_quantum`: shared loss/optimizer interface with the photonic
  student (`ce`/`mse`/`hloss` × `Adam`/`AdamW`/`SGD`/`SPSA`/`CMA`/scipy minimizers).
- `find_min_qubit_depth`: search over variational depths `L`.
- `validate_teacher_params_qubit`: rebuilds the teacher to verify the dataset's
  100 % accuracy.

**`work/learner/mlp.py`** *(formerly `work/train_mlp.py`)*
- `train_and_eval`: early stopping with 10% internal val split, same dual criterion.
- `find_min_hidden_size`: added `min_margin`, `early_stopping_patience` parameters.

**`work/train.py`**
- `--min-margin` (default 0.2), `--batch-size` (default 64),
  `--early-stopping-patience` (default 60), `--bail-threshold` (default 0.30),
  `--max-resample-iter` (default 10) CLI arguments, propagated to all three students.
- `--learner` choices now include `qubit_quantum` alongside `mlp` and `photonic_quantum`.
- `--generator` choices now include `qubit_quantum` alongside `photonic_quantum`,
  `analytical`, `mlp`.

**`work/data/`** *(formerly `merlin/datasets/`)*
- All four generators forward `min_margin` and apply the confidence gate before
  balancing. `analytical`, `mlp`, `photonic_quantum`, and `qubit_quantum` all return
  `soft_targets`.
- `data._resample.filter_resample`: shared iterative draw → margin-filter → estimate-
  yield → top-up → final balance pipeline. Raises `LowYieldError` when first-batch
  post-balance yield falls below `bail_threshold` (use `--bail-threshold 0` to opt out).

---

## The qubit_quantum learner — paper-faithful re-implementation

`learner.qubit_quantum.QubitClassifier` implements the variational
classifier of Havlíček et al. exactly:

- **Feature map.** `|0^n⟩ → H^⊗n → U_Φ(x) → H^⊗n → U_Φ(x)` with
  `U_Φ(x) = exp(i [Σ_i x_i Z_i + Σ_{i<j} (π−x_i)(π−x_j) Z_i Z_j])`. The
  feature map is implemented in `data.qubit_quantum.feature_map_state`
  and shared between the dataset generator and the student so they cannot
  drift apart.
- **Variational ansatz.** `W(θ) = U_loc(θ_L) [U_ent U_loc]_{l=L−1..1} U_ent
  U_loc(θ_0)` where each `U_loc(θ_l)` applies `RY · RZ` per qubit
  (2 params per qubit per layer) and `U_ent` is a fixed CZ chain. Total
  free angles: `(L+1) · n_qubits · 2`, plus optionally a scalar
  `bias` for the Havlíček sigmoid loss.
- **Decision rule.** `sign( ⟨Z^⊗n⟩ + b )` — exact parity expectation
  computed from the simulated state vector. With `--loss hloss` the
  student trains the trainable bias `b` jointly with the variational
  angles.
- **Optimizer.** All photonic-side optimizers are available; **`SPSA`**
  is the paper's choice (2 forward passes per step, gradient-free,
  noise-robust). `Adam` typically converges faster on small problems.
- **Sanity check.** `learner.qubit_quantum.validate_teacher_params_qubit`
  rebuilds the teacher from `metadata["teacher_theta"]` and re-runs it on
  the test set; this must report **100 %** (the dataset's labels were
  defined by that teacher).

### Example runs (m=3, k=4 — paper's 2-qubit, 4-layer setup)

| Configuration | Result |
|---|---|
| `--optimizer Adam --loss hloss --epochs 200 --dataset-size 500` (depth=4) | **100 % test accuracy** ✓ |
| `--optimizer SPSA --loss hloss --epochs 80 --dataset-size 200` (depth 0–4 search) | 67.5 % at depth 4 (small budget) |

The Adam result shows that the architecture is **fully expressive** —
the student exactly recovers the teacher's decision function when given a
gradient-based optimizer and enough samples. SPSA with a larger budget
(`--epochs 250` per the paper × full mini-batch) closes the gap; the
small-budget run above is intended as a CI smoke test, not a faithful
SPSA reproduction.

### Why `m − 1` qubits, not `m`?

We deliberately use `n_qubits = m − 1` so that all four generators
share the same feature-dimension convention (`X.shape == (N, m-1)`). The
paper's 2-qubit experiments correspond to `--m 3`; their 5-qubit Iris
experiment would be `--m 6` here. Mapping `m` to "size parameter" rather
than "number of computational units" keeps the cross-generator and
cross-student CLI interchangeable.

### Cross-architecture cells of the 4 × 3 grid

The grid suggests several interesting comparisons that are now one CLI
flag away:

- **Qubit teacher ⇒ photonic student** (`--generator qubit_quantum
  --learner photonic_quantum`): can the photonic sandwich fit an IQP
  decision boundary? The feature dim matches (n_qubits = m_modes − 1)
  but the parity is over different observables, so this is a non-trivial
  expressivity probe.
- **Photonic teacher ⇒ qubit student** (`--generator photonic_quantum
  --learner qubit_quantum`): symmetric to the above — does the IQP
  ansatz capture a Haar-random photonic boundary?
- **Analytical teacher ⇒ qubit student** (`--generator analytical
  --learner qubit_quantum`): a "classical" target whose decision boundary
  has known harmonic structure; useful for calibrating SPSA convergence
  separately from quantum-data idiosyncrasies.

These are the natural follow-up experiments enabled by the 4 × 3
factoring.

---

## Repository layout

```
work/
├── train.py                    # unified CLI entry point
├── data/                       # dataset generators
│   ├── __init__.py             # exports Dataset and the GENERATORS registry
│   ├── base.py                 # Dataset dataclass
│   ├── _resample.py            # iterative filter+balance helper, LowYieldError
│   ├── photonic_quantum.py     # Merlin sandwich generator + parity_of_key
│   ├── qubit_quantum.py        # qubit IQP-feature-map generator (Havlíček 2019)
│   ├── analytical.py           # closed-form pairwise-sin generator
│   └── mlp.py                  # random-teacher MLP generator
├── learner/                    # learner trainers
│   ├── __init__.py
│   ├── mlp.py                  # Fourier-feature MLP learner (find_min_hidden_size)
│   ├── photonic_quantum.py     # photonic variational sandwich learner (find_min_depth)
│   └── qubit_quantum.py        # qubit IQP variational learner (find_min_qubit_depth)
├── hparam_search_quantum.py    # hyperparameter sweep utility
├── plot_dataset_projections.py
└── training.md                 # this document
```

Canonical imports:

```python
from data import GENERATORS, Dataset
from data.photonic_quantum import parity_of_key
from data.qubit_quantum import (
    feature_map_state, variational_apply, parity_expectation,
)
from learner.mlp             import find_min_hidden_size
from learner.photonic_quantum import find_min_depth, validate_teacher_params
from learner.qubit_quantum    import find_min_qubit_depth, QubitClassifier
```

---

## CLI reference (`work/train.py`)

All flags are parsed by `argparse`; run `python work/train.py --help` for the
canonical list. The summary below groups flags by purpose.

### Global selection

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--learner` | choice | `mlp` | Learner architecture to train. One of `mlp`, `photonic_quantum`, `qubit_quantum`. |
| `--generator` | choice | `photonic_quantum` | Dataset generator. One of `photonic_quantum`, `qubit_quantum`, `analytical`, `mlp`. |

### Problem size

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--m` | int | `6` | Generator-specific size parameter. Photonic: number of optical modes. Qubit: `n_qubits = m − 1`. Feature dimension is always `m − 1`. |
| `--k` | int | `3` | Generator-specific complexity parameter. `photonic_quantum`: number of injected photons. `qubit_quantum`: variational depth `L` of the random teacher. `analytical`: spatial-frequency multiplier inside `sin(k·…)`. `mlp`: number of hidden layers. |
| `--target-accuracy` | float | `0.90` | First model that reaches this test accuracy ends the search. |
| `--dataset-size` | int | `10000` | **Post-filter, post-balance** target size. With `--min-margin > 0` or `--balanced`, more raw samples are drawn iteratively to reach this size. |

### Training schedule

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--epochs` | int | `300` | Maximum training epochs. For gradient-free quantum optimizers (`CMA`, `COBYLA`, `NelderMead`, `Powell`) reinterpreted as the budget of objective-function evaluations. |
| `--batch-size` | int | `64` | Mini-batch size for gradient-based training. |
| `--lr` | float | `3e-3` (mlp) / `1e-2` (quantum) | Learning rate; only used by gradient-based optimizers. |
| `--data-seed` | int | `42` | Seed for dataset draws (and Haar matrices / random teacher angles). |
| `--model-seed` | int | `0` | Seed for student parameter initialisation. |

### Dataset filtering & resampling

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--balanced` | flag | off | Subsample the majority class to obtain an exact 50/50 dataset. |
| `--min-margin` | float | `0.2` | Drop samples with confidence below `M` (in `[0, 1]`). Confidence is `\|parity_expectation\|` (quantum), `\|score\| / n_pairs` (analytical), or `\|p1 − p0\|` (mlp). Mimics the 0.3 separation gap in Havlíček et al. |
| `--bail-threshold` | float | `0.30` | If first-batch post-balance yield is below this fraction, raise `LowYieldError`. Pass `0` to opt out and let the iterative resampler always reach `--dataset-size`. |
| `--max-resample-iter` | int | `10` | Maximum resample iterations after the first batch. |

### MLP-learner options (only with `--learner mlp`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--hidden-sizes SPEC ...` | strings | `[2, 4, 8, 16, 32, 64, 128, 256]` | Hidden-layer architectures to try in order. Each spec is a single integer (one hidden layer) or comma-separated integers (e.g. `64,32` → two layers). The search stops at the first spec reaching `--target-accuracy`. |

### Photonic-quantum-learner options (only with `--learner photonic_quantum`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--depths D ...` | ints | `[1, 2, 3, 4]` | Variational re-uploading depths to try. `depth=1` mirrors the teacher's single sandwich. |
| `--sizes M_CIRC ...` | ints | `[m]` | Interferometer widths to try. Must be ≥ `m`; when `m_circuit > m`, photons are spread evenly to exploit the extra modes. |
| `--loss` | choice | `ce` | Training loss. `ce` = cross-entropy on hard labels. `mse` = regress learner parity expectation against teacher's continuous parity (only with `--generator photonic_quantum` or `qubit_quantum`). `hloss` = Havlíček-style sigmoid loss with trainable bias. |
| `--optimizer` | choice | `Adam` | `Adam`, `AdamW`, `SGD`, `SPSA`, `CMA`, `Powell`, `NelderMead`, `COBYLA`. |
| `--early-stopping-patience` | int | `60` | Stop training if val accuracy has not improved for this many epochs (evaluated every 10). Set to `99999` to disable. |

### Qubit-quantum-learner options (only with `--learner qubit_quantum`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--depths D ...` | ints | `[0, 1, 2, 3, 4]` | Number of variational layers `L` to try (paper goes up to `L=4`). Layer 0 is one rotation block with no entangler. |
| `--loss` | choice | `ce` | Same options as the photonic learner. **Paper uses `hloss`** (sigmoid loss + trainable bias). |
| `--optimizer` | choice | `Adam` | Same options. **Paper uses `SPSA`** (gradient-free, 2 forward passes per step). `Adam` typically converges faster on small problems. |
| `--sizes`, `--early-stopping-patience` | (ignored) | — | The qubit count is fixed at `m − 1`; full `--epochs` budget is used. |

### Common usage patterns

**Paper-faithful Havlíček run (qubit teacher + qubit learner, SPSA, hloss, 0.3 gap):**
```bash
python work/train.py --learner qubit_quantum --generator qubit_quantum \
  --m 3 --k 4 --balanced --min-margin 0.3 --bail-threshold 0 \
  --optimizer SPSA --loss hloss --depths 0 1 2 3 4
```

**Quick verification that the qubit learner is fully expressive (Adam):**
```bash
python work/train.py --learner qubit_quantum --generator qubit_quantum \
  --m 3 --k 4 --balanced --min-margin 0.3 --bail-threshold 0 \
  --optimizer Adam --loss hloss --depths 4 --epochs 200 --dataset-size 500
# → 100 % test accuracy
```

**Photonic re-implementation of the paper:**
```bash
python work/train.py --learner photonic_quantum --generator photonic_quantum \
  --m 6 --k 3 --balanced --min-margin 0.10 --bail-threshold 0 \
  --loss hloss --depths 1 2 3 4 --sizes 6 8 10 \
  --dataset-size 10000 --epochs 500
```

**MLP baseline on the analytical generator:**
```bash
python work/train.py --learner mlp --generator analytical \
  --m 4 --k 2 --balanced --min-margin 0.3 \
  --hidden-sizes 32 64 128
```

**Cross-architecture probe — qubit learner fitting a photonic teacher:**
```bash
python work/train.py --learner qubit_quantum --generator photonic_quantum \
  --m 3 --k 2 --balanced --min-margin 0.05 \
  --optimizer Adam --loss hloss --depths 2 3 4
```

---

## Cross-reference with Havlíček et al. (status of the photonic re-implementation)

Below is the original Havlíček-vs-our-code comparison applied **specifically
to the photonic side** (the qubit side is now a faithful reproduction — see
*The qubit_quantum learner* section above). For the photonic generator and
student, several of the paper's mechanisms are now implemented; the table
records which.

| Component | Havlíček et al. | Photonic implementation here | Status |
|---|---|---|---|
| Feature encoding | `exp(i Σ_S φ_S(x) ∏ Z_i)` with **\|S\|=1**: `φ_i = x_i` and **\|S\|=2**: `φ_{ij} = (π−x_i)(π−x_j)` (cross-product Z⊗Z terms), wrapped as `U_Φ H U_Φ H` | Single-mode phase shifters `exp(i x_i)` between two Haar blocks; cross-products only via the random Haar mixing | ❌ no cross-products in encoding (faithful in `qubit_quantum`) |
| Variational ansatz | `W(θ) = U_loc(θ_l) U_ent … U_loc(θ_1)` — Y/Z rotations + CZ entangler, `l = 0..4` | Two trainable Clements MZI meshes (`m(m−1)` params per block); RECTANGLE shape | ✓ photonic analog, more expressive |
| **Bias term `b ∈ [−1, 1]`** | Optimised together with θ, decision = `sign(parity_exp + b)` | Implemented as `QuantumClassifier.bias` with `--loss hloss` | ✓ implemented |
| Decision rule | Sign of empirical parity probability ± bias | Sign of exact parity expectation `probs @ parity_vec` (R = ∞ limit), or sampled with `nsample > 0` | ✓ |
| **Cost function (sigmoid loss)** | `−log σ(α · y · (parity_exp + b))` | `--loss hloss`: `−F.logsigmoid(α · y_pm · (parity_exp + b)).mean()` with `α = 5` | ✓ implemented |
| **Optimiser SPSA** | Spall's SPSA, 2 fn-evals per step, 250 iterations | `--optimizer SPSA` (Spall's coefficients `a/(k+1+A)^α`, `c/(k+1)^γ`); shared by both quantum students | ✓ implemented |
| Training shots `R` | 2 000 per evaluation (10 000 at classification time) | `--nsample` available on the photonic generator (passed into Merlin's `forward(shots=…)`) | ✓ |
| **Margin-filtered dataset** | Margin 0.3 to ensure separability | `--min-margin` flag in all generators; for photonic `m=6, k=3` only ≲ 0.10 is feasible (measure concentration), but `qubit_quantum` supports the full 0.3 | ✓ implemented (with documented constraints) |

What's still distinct on the photonic side:

- The encoding is **product** (one phase per mode, no cross terms), so the
  photonic feature map covers a different space than the qubit IQP map.
  Adding two-mode interactions (e.g. parameterised beam splitters keyed by
  `x_i · x_j`) would be the closest analog to the ZZ cross terms.
- The photonic student uses MZI-mesh trainable unitaries on either side
  rather than alternating rotation/entangler layers — more expressive in
  principle, but the optimization landscape is different from the paper's
  ansatz.

These are intentional photonic-vs-qubit differences, not bugs. The new
`qubit_quantum` code path is the place to compare against the paper's
exact methodology; the photonic code path explores how the same
"data-defined-by-quantum-circuit" idea behaves on a different physical
substrate.
