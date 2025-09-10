# QRKD: Quantum Relational Knowledge Distillation (Reproduction)

This folder contains the reproduction of the paper:

- Title: Quantum Relational Knowledge Distillation
- arXiv: https://arxiv.org/abs/2508.13054

## Overview

This reproduction follows the guidelines from the repository README and the MerLin documentation. It includes a minimal, testable implementation and an exploratory notebook.

## Contents

- `implementation.py`: Core implementation for the QRKD reproduction.
- `notebook.ipynb`: Interactive notebook to explore the method and reproduce key figures.
- `data/`: Datasets and preprocessing scripts.
- `results/`: Generated figures and metrics.
- `tests/`: Sanity checks and minimal validation tests.

## How to run
Run everything from this `QRKD/` directory with a local virtual environment.

Setup:

```bash
cd QRKD
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Command-line usage (train one model at a time):

```bash
# Train a teacher for 1 epoch and save it under models/
python implementation.py --task teacher --dataset mnist --epochs 1 --seed 0 --save-dir models

# Train a student with KD using the saved teacher
python implementation.py --task student_kd --dataset mnist --epochs 1 --seed 0 \
	--teacher-path models/mnist_teacher_seed0_e1.pt --save-dir models

# Other variants
# - student_scratch: student trained only with cross-entropy (no distillation)
# - student_rkd: student trained with RKD (distance+angle) only
# - student_qrkd: student trained with KD + RKD (classical part of QRKD)

# Optional: reduce logs
python implementation.py --task student_qrkd --epochs 5 --quiet --teacher-path models/mnist_teacher_seed0_e5.pt

# Quick check runs (faster dev loop)
# Use --checkrun to limit each epoch to 50 batches.
# Saved files get a suffix _chk (e.g., mnist_teacher_seed0_e1_chk.pt)
python implementation.py --task teacher --epochs 1 --checkrun
python implementation.py --task student_scratch --epochs 1 --checkrun \
	--teacher-path models/mnist_teacher_seed0_e1_chk.pt
```

Notes:
- `student_scratch` uses only the task loss (CrossEntropy) without any distillation.
- `student_kd` adds a KL-based knowledge distillation term on logits (no RKD).

Notebook:

```bash
jupyter notebook notebook.ipynb
```

## Status

- [ ] WIP: initial scaffolding
- [ ] Baseline results
- [ ] Parity with paper figures
- [ ] Extensions and ablations
