#!/usr/bin/env python3
"""QLSTM reproduction CLI entrypoint.

"""
from __future__ import annotations

import argparse
import json
import os

from lib.utils import set_seed

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="QLSTM reproduction CLI (skeleton)")
    ap.add_argument("--task", choices=["baseline_lstm", "qlstm"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", type=str, default="models")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)



if __name__ == "__main__":
    main()
