import os
import sys

# Ensure imports resolve to the top-level package at <repo_root>/QRKD,
# not the nested folder <repo_root>/QRKD/QRKD/ (which may exist for data).
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
cwd = os.getcwd()

new_path = [REPO_ROOT]
for p in sys.path:
    # Drop the empty entry ('') that points to CWD and any explicit CWD entry
    if p in ("", cwd):
        continue
    # Keep other entries
    if p != REPO_ROOT:
        new_path.append(p)
sys.path[:] = new_path
