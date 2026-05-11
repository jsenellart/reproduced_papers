"""Train a 2-layer MLP on the quantum parity dataset and find the minimal
hidden size to reach a target accuracy."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import GENERATORS


# ---------------------------------------------------------------------------
# Teacher validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_teacher_params(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    metadata: dict,
) -> float:
    """Reload the teacher MLP from stored state_dict and measure accuracy.

    With the exact teacher parameters the student is trying to mimic,
    accuracy must be 100 % — this is a pre-training sanity check.
    """
    fourier_order = metadata["teacher_fourier_order"]
    hidden_size   = metadata["hidden_size"]
    n_layers      = metadata["n_hidden_layers"]
    m             = metadata["m"]
    d             = m - 1
    feat_size     = 2 * fourier_order * d

    # Reconstruct teacher architecture (must match generate_mlp_teacher)
    layers: list[nn.Module] = []
    in_size = feat_size
    for _ in range(n_layers):
        layers += [nn.Linear(in_size, hidden_size, bias=False), nn.Tanh()]
        in_size = hidden_size
    layers.append(nn.Linear(in_size, 2, bias=False))
    teacher = nn.Sequential(*layers)
    teacher.load_state_dict(metadata["teacher_state_dict"])
    teacher.eval()

    # Compute Fourier features (same as in generate_mlp_teacher)
    parts = []
    for j in range(1, fourier_order + 1):
        parts.append(torch.sin(j * X_test))
        parts.append(torch.cos(j * X_test))
    phi = torch.cat(parts, dim=-1)

    y_pred = teacher(phi).argmax(dim=-1)
    return (y_pred == y_test.long()).float().mean().item()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _normalize_hidden_layers(hidden_layers: int | Sequence[int]) -> list[int]:
    if isinstance(hidden_layers, int):
        normalized = [hidden_layers]
    else:
        normalized = list(hidden_layers)

    if not normalized:
        raise ValueError("hidden_layers must contain at least one layer size")
    if any(size <= 0 for size in normalized):
        raise ValueError("hidden layer sizes must be positive")

    return normalized


HiddenLayerSpec = int | tuple[int, ...]

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: int | Sequence[int],
        fourier_order: int = 3,
    ) -> None:
        super().__init__()
        # Fourier features: sin(k*x), cos(k*x) for k=1..fourier_order
        self.fourier_order = fourier_order
        feat_size = 2 * fourier_order * input_size

        layers = []
        in_features = feat_size
        for hidden_size in _normalize_hidden_layers(hidden_layers):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) angles in [0, 2π]
        features = []
        for k in range(1, self.fourier_order + 1):
            features.append(torch.sin(k * x))
            features.append(torch.cos(k * x))
        return self.net(torch.cat(features, dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_eval(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_layers: int | Sequence[int],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    early_stopping_patience: int = 60,
) -> float:
    """Train one MLP and return test accuracy."""
    torch.manual_seed(seed)

    # labels are 0/1 (or +1/-1 legacy), convert to 0/1 for CrossEntropyLoss
    y_tr = ((y_train + 1) // 2).long()
    y_te = ((y_test + 1) // 2).long()

    # 10% val split for early stopping
    n_val = max(1, len(X_train) // 10)
    X_val, y_val_es = X_train[-n_val:], y_tr[-n_val:]
    X_tr2, y_tr2 = X_train[:-n_val], y_tr[:-n_val]

    loader = DataLoader(
        TensorDataset(X_tr2, y_tr2),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP(X_train.shape[1], hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    import copy
    best_val_acc = -1.0
    best_train_loss = float("inf")
    best_state = None
    no_improve_epochs = 0

    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val).argmax(1) == y_val_es).float().mean().item()
            model.train()

            val_improved  = val_acc    > best_val_acc
            loss_improved = epoch_loss < best_train_loss

            if val_improved:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            if loss_improved:
                best_train_loss = epoch_loss

            if not val_improved and not loss_improved:
                no_improve_epochs += 10
            else:
                no_improve_epochs = 0

            if no_improve_epochs >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_te).float().mean().item()

    return accuracy


# ---------------------------------------------------------------------------
# Search for minimal hidden size
# ---------------------------------------------------------------------------

def find_min_hidden_size(
    m: int,
    k: int,
    generator: str = "quantum",
    target_accuracy: float = 0.90,
    dataset_size: int = 2000,
    hidden_sizes: list[HiddenLayerSpec] | None = None,
    test_fraction: float = 0.20,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    data_seed: int = 42,
    model_seed: int = 0,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_resample_iter: int = 10,
    early_stopping_patience: int = 60,
) -> None:
    """Find the first hidden-layer configuration reaching the target accuracy.

    generator   One of the keys in datasets.GENERATORS: 'quantum',
                'analytical', or 'mlp'.
    hidden_sizes can contain either integers for one hidden layer or tuples
    for deeper architectures, for example [64, (64, 32), (128, 64, 32)].
    """
    if hidden_sizes is None:
        hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256]

    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator!r}. Choose from {list(GENERATORS)}.")

    margin_str = f"  min_margin={min_margin}" if min_margin > 0.0 else ""
    print(f"Generating dataset  generator={generator}  m={m}  k={k}  size={dataset_size}"
          f"  balanced={balanced}{margin_str} ...")
    ds = GENERATORS[generator](size=dataset_size, m=m, k=k, seed=data_seed,
                               balanced=balanced, min_margin=min_margin,
                               bail_threshold=bail_threshold,
                               max_iter=max_resample_iter)

    # 80/20 train/test split
    n_test = int(len(ds.X) * test_fraction)
    n_train = len(ds.X) - n_test
    train_ds, test_ds = random_split(
        TensorDataset(ds.X, ds.y),
        [n_train, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    X_train, y_train = train_ds[:][0], train_ds[:][1]
    X_test, y_test = test_ds[:][0], test_ds[:][1]

    print(f"Train: {n_train}  Test: {n_test}")
    print(f"Target accuracy: {target_accuracy:.0%}")

    # Pre-training sanity check: teacher MLP with exact weights must give 100 %
    if "teacher_state_dict" in ds.metadata:
        teacher_acc = validate_teacher_params(X_test, y_test, ds.metadata)
        print(f"Teacher (exact params) validation: {teacher_acc:.2%}  ← should be 100%")

    print(f"\n{'hidden_layers':>20}  {'accuracy':>10}")
    print("-" * 34)

    found: HiddenLayerSpec | None = None
    for h in hidden_sizes:
        label = str(h)
        acc = train_and_eval(
            X_train, y_train, X_test, y_test,
            hidden_layers=h,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=model_seed,
            early_stopping_patience=early_stopping_patience,
        )
        marker = " <-- first hit" if found is None and acc >= target_accuracy else ""
        print(f"{label:>20}  {acc:>10.2%}{marker}")
        if found is None and acc >= target_accuracy:
            found = h
            break

    print()
    if found is not None:
        print(f"Minimal hidden size to reach {target_accuracy:.0%}: {found}")
    else:
        print(f"No hidden size in {hidden_sizes} reached {target_accuracy:.0%}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the minimal hidden-layer architecture reaching a target accuracy."
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default="quantum",
        help="Dataset generation strategy (default: quantum)",
    )
    parser.add_argument("--m", type=int, default=6, help="Number of optical modes")
    parser.add_argument("--k", type=int, default=2, help="Complexity parameter (photons / depth / frequency)")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--dataset-size", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=0)
    cli = parser.parse_args()

    find_min_hidden_size(
        m=cli.m,
        k=cli.k,
        generator=cli.generator,
        target_accuracy=cli.target_accuracy,
        dataset_size=cli.dataset_size,
        hidden_sizes=[(64, 64, 64, 64)],
        epochs=cli.epochs,
        lr=cli.lr,
        data_seed=cli.data_seed,
        model_seed=cli.model_seed,
    )
