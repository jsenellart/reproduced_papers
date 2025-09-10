# QRKD: Quantum Random Kitchen Directions (Reproduction)

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

Quick run (short training for a fast check):

```bash
QRKD_TEACHER_EPOCHS=1 QRKD_EPOCHS=1 python -c "from QRKD.implementation import run; run()"
```

Notebook:

```bash
jupyter notebook notebook.ipynb
```

## Status

- [ ] WIP: initial scaffolding
- [ ] Baseline results
- [ ] Parity with paper figures
- [ ] Extensions and ablations
