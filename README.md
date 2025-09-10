# MerLin Reproduced Papers

## About this repository


This repository contains implementations and resources for reproducing key quantum machine learning papers, with a focus on photonic and optical quantum computing.

It is part of the main MerLin project: [https://github.com/merlinquantum/merlin](https://github.com/merlinquantum/merlin)
and complements the online documentation available at:

[https://merlinquantum.ai/research/reproduced_papers.html](https://merlinquantum.ai/research/reproduced_papers.html)

Each paper reproduction is designed to be accessible, well-documented, and easy to extend. Contributions are welcome!

## Installation

Each paper reproduction has its own `requirements.txt` inside its folder. Navigate into that folder and install dependencies there.

Example (QRKD):

```bash
cd QRKD
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## How to contribute a reproduced paper

We encourage contributions of new quantum ML paper reproductions. Please follow the guidelines below:

### Recommended structure for a reproduction

```
paper_reproduction/
├── README.md             # Paper overview and results
├── implementation.py     # Core implementation
├── notebook.ipynb        # Interactive exploration of key concepts
├── data/                 # Datasets and preprocessing
├── results/              # Figures and analysis
└── tests/                # Validation tests
```

### Submission process

1. **Propose** the paper in our [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions)
2. **Implement** using the repository tools, following the structure above
3. **Validate** results against the original paper
4. **Document** in Jupyter notebook format
5. **Submit** a pull request with the complete reproduction folder

### Contribution requirements

- High-impact quantum ML papers (>50 citations preferred)
- Photonic/optical quantum computing focus
- Implementable with current repository features
- Clear experimental validation

### Recognition

Contributors are recognized in:
- Paper reproduction documentation
- MerLin project contributors list
- Academic citations in MerLin publications

## Code Style and Quality

This repository uses [Ruff](https://docs.astral.sh/ruff/) for consistent code formatting and linting across all paper implementations.

### Usage

**Check code style:**
```bash
ruff check .
```

**Format code:**
```bash
ruff format .
```

**Install pre-commit hooks (recommended):**
```bash
pip install pre-commit
pre-commit install
```

### Configuration

- Code style rules are defined in `pyproject.toml`
- GitHub Actions automatically check all PRs and pushes
- Pre-commit hooks run ruff automatically before commits