"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
from traser.dynamics.initialization import initialize, seed_for


def _seed(dataset: str, stage: int, origin: int, step: int, replicate: int = 0) -> int:
    stream = {1: "initialization", 2: "development", 3: "training_innovation", 4: "test"}[stage]
    return seed_for(dataset, stream, origin, step, replicate)


def _initial(
    cloud: np.ndarray, budget: int, dataset: str, origin: int, replicate: int
) -> np.ndarray:
    return initialize(cloud, budget, _seed(dataset, 1, origin, 0, replicate))[0]
