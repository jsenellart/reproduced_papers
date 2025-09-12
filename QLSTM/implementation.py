#!/usr/bin/env python3
"""QLSTM reproduction CLI.

Trains classical LSTM or QLSTM on toy sequence generators.

Adapted from Quantum_Long_Short_Term_Memory (MIT License), refactored into
modular components (dataset/model/rendering) for clarity & reproducibility.
"""
from __future__ import annotations

import argparse, os, json, time
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn

from lib.dataset import data as data_factory
from lib.model import build_model
from lib.rendering import save_losses_plot, save_simulation_plot, save_pickle


@dataclass
class ExpConfig:
    generator: str = "damped_shm"
    model_type: str = "qlstm"  # or 'lstm'
    seq_length: int = 4
    hidden_size: int = 5
    vqc_depth: int = 4
    batch_size: int = 10
    epochs: int = 50
    learning_rate: float = 0.01
    train_split: float = 0.67
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir: str | None = None
    fmt: str = "png"
    save_only_last: bool = True


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(cfg: ExpConfig):
    set_seed(cfg.seed)
    if cfg.generator == 'csv' and hasattr(cfg, 'csv_path') and cfg.csv_path:
        gen = data_factory.get('csv', path=cfg.csv_path)
    else:
        gen = data_factory.get(cfg.generator)
    x, y = gen.get_data(cfg.seq_length)
    n_train = int(cfg.train_split * len(x))
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    x_train_in = x_train.unsqueeze(2)
    x_test_in = x_test.unsqueeze(2)
    model = build_model(cfg.model_type, 1, cfg.hidden_size, cfg.vqc_depth, 1).to(cfg.device).double()
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg.learning_rate)
    mse = nn.MSELoss()
    train_losses=[]; test_losses=[]; epochs_list=[]
    exp_dir = cfg.exp_dir or f"experiments/{cfg.model_type.upper()}_TS_MODEL_{cfg.generator.upper()}"
    os.makedirs(exp_dir, exist_ok=True)
    ts = time.strftime("NO%Y%m%d%H%M%S")
    base = os.path.basename(exp_dir)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(x_train_in.size(0))
        bl=[]
        for i in range(0, x_train_in.size(0), cfg.batch_size):
            idx = perm[i:i+cfg.batch_size]
            xb = x_train_in[idx].to(cfg.device)
            yb = y_train[idx].to(cfg.device)
            pred,_ = model(xb)
            pred_last = pred.transpose(0,1)[-1].view(-1)
            loss = mse(pred_last, yb)
            opt.zero_grad(); loss.backward(); opt.step(); bl.append(loss.item())
        tr_loss = float(sum(bl)/len(bl));
        with torch.no_grad():
            model.eval(); ptest,_ = model(x_test_in.to(cfg.device))
            te_loss = mse(ptest.transpose(0,1)[-1].view(-1), y_test.to(cfg.device)).item()
        train_losses.append(tr_loss); test_losses.append(te_loss); epochs_list.append(epoch+1)
        if (epoch+1)%10==0 or epoch==0:
            print(f"Epoch {epoch+1}: train {tr_loss:.6f} test {te_loss:.6f}")
        last = (epoch==cfg.epochs-1)
        if last or not cfg.save_only_last:
            save_losses_plot(train_losses, exp_dir, prefix=f"{base}_NO_1_Epoch_{epoch+1}_train")
            # simple inference over full series
            full_pred,_ = model(x.unsqueeze(2).to(cfg.device))
            full_last = full_pred.transpose(0,1)[-1].view(-1).detach().cpu().numpy()
            save_simulation_plot(y.detach().cpu().numpy(), full_last, exp_dir, prefix=f"{base}_NO_1_Epoch_{epoch+1}_simulation")
    torch.save(model.state_dict(), os.path.join(exp_dir, f"{base}_NO_1_Epoch_{cfg.epochs}_{ts}_torch_model.pth"))
    save_pickle(train_losses, exp_dir, f"{base}_NO_last_TRAINING_LOSS")
    save_pickle(test_losses, exp_dir, f"{base}_NO_last_TESTING_LOSS")
    with open(os.path.join(exp_dir, "config.json"),"w") as f: json.dump(asdict(cfg), f, indent=2)
    return model, train_losses, test_losses, exp_dir, ts, epochs_list[-1]


def main():
    ap = argparse.ArgumentParser(description="QLSTM vs LSTM trainer")
    ap.add_argument("--model", choices=["qlstm","lstm"], default="qlstm")
    ap.add_argument("--generator", default="damped_shm", help="sin | damped_shm | logsine | ma_noise | csv")
    ap.add_argument("--csv-path", type=str, default=None, help="Path to CSV file when --generator csv")
    ap.add_argument("--seq-length", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=5)
    ap.add_argument("--vqc-depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--train-split", type=float, default=0.67)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp-dir", type=str, default=None)
    ap.add_argument("--fmt", choices=["png","pdf"], default="png")
    ap.add_argument("--save-all", action="store_true")
    args = ap.parse_args()
    cfg = ExpConfig(generator=args.generator, model_type=args.model, seq_length=args.seq_length, hidden_size=args.hidden_size,
                    vqc_depth=args.vqc_depth, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr,
                    train_split=args.train_split, seed=args.seed, exp_dir=args.exp_dir, fmt=args.fmt, save_only_last=not args.save_all)
    # attach csv path dynamically (not in dataclass to avoid clutter)
    if args.csv_path:
        setattr(cfg, 'csv_path', args.csv_path)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
