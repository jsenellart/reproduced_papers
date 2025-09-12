from __future__ import annotations
"""Rendering utilities for QLSTM reproduction.

Adapted from Quantum_Long_Short_Term_Memory (MIT License) with simplifications.
"""
import os, pickle, time
from typing import Sequence
import matplotlib.pyplot as plt

def timestamp():
    return time.strftime('%Y%m%d-%H%M%S')

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def save_losses_plot(losses:Sequence[float], out_dir:str, prefix:str='loss'):
    ensure_dir(out_dir)
    plt.figure(figsize=(5,3))
    plt.plot(losses)
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.tight_layout()
    fname = os.path.join(out_dir, f'{prefix}_plot_{timestamp()}.png')
    plt.savefig(fname); plt.close(); return fname

def save_simulation_plot(y_true, y_pred, out_dir:str, prefix:str='simulation'):
    ensure_dir(out_dir)
    plt.figure(figsize=(6,3))
    plt.plot(y_true, label='target')
    plt.plot(y_pred, label='pred')
    plt.legend(); plt.tight_layout()
    fname = os.path.join(out_dir, f'{prefix}_{timestamp()}.png')
    plt.savefig(fname); plt.close(); return fname

def save_pickle(obj, out_dir:str, name:str):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f'{name}_{timestamp()}.pkl')
    with open(path,'wb') as f: pickle.dump(obj,f)
    return path
