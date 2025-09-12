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
- `scripts/run_table1_sweep.py`: Multi-seed automation (train all teachers, pick best teacher, then train Scratch/KD/RKD; reports mean±std Table 1 metrics).
- `scripts/plot_curves.py`: Aggregate per-epoch loss & test accuracy across seeds (Teacher, Scratch, KD, RKD, optional QRKD) + shaded std bands.
- `scripts/evaluate_table1.py`: Single-seed Table 1 style evaluation (train/test, gaps, gains).

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
python implementation.py --task student_scratch --epochs 1 --checkrun
```

Notes:
- `student_scratch` uses only the task loss (CrossEntropy) without any distillation.
- `student_kd` adds a KL-based knowledge distillation term on logits (no RKD).

Notebook:

```bash
jupyter notebook notebook.ipynb
```

## Evaluation (Table 1)

Un script autonome calcule les métriques à la manière de la Table 1 (MNIST):
- Train Acc, Test Acc
- Acc. Gap = Train − Test
- T&S Gap = Teacher(Test) − Student(Test)
- Dist. Gain = Student(Test) − Scratch(Test)

Exécution (depuis `QRKD/`):

```bash
# Exemple avec checkpoints entraînés 10 epochs
.venv/bin/python scripts/evaluate_table1.py \
	--teacher models/mnist_teacher_seed0_e10.pt \
	--scratch models/mnist_student-scratch_seed0_e10.pt \
	--kd      models/mnist_student-kd_seed0_e10.pt \
	--rkd     models/mnist_student-rkd_seed0_e10.pt \
	--qrkd    models/mnist_student-qrkd_seed0_e10.pt

# Option: préciser l’emplacement des données
.venv/bin/python scripts/evaluate_table1.py \
	--teacher models/mnist_teacher_seed0_e10.pt \
	--data-dir QRKD/data
```

Le script charge les modèles, évalue accuracies sur train/test et affiche un tableau compact.

### Multi-seed sweep & plots

Pour reproduire Table 1 (moyenne ± écart-type) et préparer les courbes (style Figure 5):

```bash
# Multi-seed (ex: 5 seeds, 10 epochs) – lance depuis QRKD/
.venv/bin/python scripts/run_table1_sweep.py --epochs 10 --seeds 0 1 2 3 4

# Générer les courbes (loss & test acc avec bandes ±std)
.venv/bin/python scripts/plot_curves.py --dataset mnist --epochs 10 --save-dir models --out-dir results
```

Les figures produites:
- `results/loss_vs_epoch.png`
- `results/testacc_vs_epoch.png`

### RKD (Distance & Angle) – définition utilisée

La ré-implémentation suit fidèlement le référentiel officiel RKD:
- Distance: normalisation par la moyenne des distances > 0 (excluant diagonale) pour chaque lot, SmoothL1 entre matrices normalisées.
- Angle: tenseur des différences pairwise (N×N×D), normalisation L2, produit matriciel (`bmm`) pour obtenir toutes les relations angulaires, aplatis puis SmoothL1.

Formules (simplifiées):
```
d_ij = ||f_i - f_j||_2
\bar d = mean_{i≠j}(d_ij)  ;  D = d / \bar d
Loss_dist = SmoothL1( D_student , D_teacher )

Δ_ij = f_i - f_j;   u_ij = Δ_ij / ||Δ_ij||_2
Angle_ijkl (implémenté via (u * u^T)) → vecteur aplati
Loss_angle = SmoothL1( A_student , A_teacher )

Loss_RKD = w_dr * Loss_dist + w_ar * Loss_angle
```

Poids par défaut (classique): `kd=0.5, dr=0.1, ar=0.1` pour la combinaison (KD + RKD) — ajustables via `QRKDWeights`.

## Status

- [x] Scaffolding & baselines (Teacher, Scratch, KD, RKD)
- [x] Multi-seed Table 1 reproduction (mean±std + gaps/gains)
- [x] Courbes loss & accuracy avec bandes d'écart-type
- [x] Mise à jour implémentation RKD conforme repo officiel
- [ ] Intégration composante quantique (prochaine étape)
- [ ] Ablations et hyperparam tuning avancé
