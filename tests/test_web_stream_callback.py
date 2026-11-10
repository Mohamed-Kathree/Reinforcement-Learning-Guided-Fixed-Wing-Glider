"""
test_web_stream_callback.py
===========================
The dashboard callback must serialise whatever SB3 logs. SB3 2.8 logs
train/approx_kl as np.float32, which plain json.dumps rejects; that crashed
the first V20 training run after one PPO iteration.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.callbacks.web_stream_callback import dumps_record  # noqa: E402


def test_numpy_scalars_serialise():
    record = {
        "approx_kl": np.float32(0.0123),
        "clip_fraction": np.float64(0.2),
        "n_updates": np.int64(10),
        "stage": 0,
        "outcome_counts": {"success": np.int32(3), "crash": None},
        "vec": np.array([1.0, 2.0], dtype=np.float32),
    }
    out = json.loads(dumps_record(record))
    assert out["approx_kl"] == float(np.float32(0.0123))
    assert out["n_updates"] == 10 and out["outcome_counts"]["success"] == 3
    assert out["vec"] == [1.0, 2.0]


def test_sb3_logged_approx_kl_type_is_handled():
    """Exactly what PPO.train() records: np.mean of float32 arrays."""
    approx_kl = np.mean([np.float32(0.01), np.float32(0.02)])
    assert isinstance(approx_kl, np.float32)
    json.loads(dumps_record({"approx_kl": approx_kl}))
