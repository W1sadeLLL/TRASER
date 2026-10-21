"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

import numpy as np


def weighted_batch_moments(x, weights):
    """Unnormalised moments; integer occurrence weights equal expanded rows."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    if (
        x.ndim != 2
        or w.shape != (len(x),)
        or (not np.isfinite(x).all())
        or (not np.isfinite(w).all())
        or np.any(w <= 0)
    ):
        raise ValueError("Invalid weighted PCA batch")
    sx = np.sum(x.astype(np.float64) * w[:, None], axis=0)
    xw = x * np.sqrt(w.astype(np.float32))[:, None]
    return (sx, (xw.T @ xw).astype(np.float64))
