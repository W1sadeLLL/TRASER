"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp

STREAM_CODES = {"initialization": 1, "development": 2, "training_innovation": 3, "test": 4}


def seed_for(dataset: str, stream: str, window: int, step: int = 0, replicate: int = 0) -> int:
    platform = {"blm_x": 1, "hk_x": 2, "hk_weibo": 3}[dataset]
    return int(
        np.random.SeedSequence(
            (platform, STREAM_CODES[stream], window, step, replicate)
        ).generate_state(1)[0]
    )


def initialize(points: np.ndarray, budget: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or len(points) == 0 or (not np.isfinite(points).all()):
        raise ValueError("Initialization requires a nonempty finite cloud")
    rng = np.random.default_rng(seed)
    n = len(points)
    if n >= budget:
        indices = rng.choice(n, budget, replace=False)
    else:
        q, remainder = divmod(budget, n)
        indices = np.concatenate(
            (np.tile(np.arange(n, dtype=np.int64), q), rng.choice(n, remainder, replace=False))
        )
        rng.shuffle(indices)
    return (points[indices], indices)
