"""
config.py
=========
Path wiring between the dashboard backend and the RL glider research repo.

The backend never reimplements simulation code -- it imports the real
env/sim/baseline/training packages from GLIDER_REPO_ROOT and shells out to
the real CLIs (validate_glide.py, pytest, training.train). Those imports and
subprocess calls both use the RL repo's own virtualenv interpreter
(GLIDER_PYTHON below), not the machine's system Python -- the venv is the
only environment on this machine with jsbsim/gymnasium/stable-baselines3
installed, so calling system Python here would fail exactly like a fresh
checkout does before `pip install -r requirements.txt`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = BACKEND_DIR.parent  # backend/ lives inside the RL repo itself

GLIDER_REPO_ROOT = Path(os.environ.get("GLIDER_REPO_ROOT", str(REPO_ROOT_DEFAULT))).resolve()
GLIDER_PYTHON = GLIDER_REPO_ROOT / "venv" / "Scripts" / "python.exe"

DATA_DIR = BACKEND_DIR.parent / "data"
EPISODES_DIR = DATA_DIR / "episodes"
# One stream file (<ISO8601>_seed<N>.jsonl) + one sibling metadata file
# (<same stem>.meta.json) per training run -- V16 Phase A (§A5). Replaces
# the old single, run-truncated TRAIN_STREAM_PATH: every run used to
# overwrite its predecessor, destroying run history. See
# backend/callbacks/web_stream_callback.py and
# backend/routers/training.py's GET /api/training/runs.
RUNS_DIR = DATA_DIR / "runs"

EPISODES_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

if str(GLIDER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GLIDER_REPO_ROOT))


def repo_ok() -> bool:
    """True if GLIDER_REPO_ROOT looks like the real RL glider repo."""
    markers = [
        GLIDER_REPO_ROOT / "env" / "glider_env.py",
        GLIDER_REPO_ROOT / "sim" / "jsbsim_fdm.py",
        GLIDER_REPO_ROOT / "validate_glide.py",
    ]
    return GLIDER_REPO_ROOT.is_dir() and all(m.is_file() for m in markers)


def python_ok() -> bool:
    """True if the wrapped venv interpreter (GLIDER_PYTHON) exists."""
    return GLIDER_PYTHON.is_file()
