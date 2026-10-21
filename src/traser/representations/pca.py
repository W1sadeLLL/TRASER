"""Weighted training PCA rebuilding from an explicit, isolated embedding cache."""

import numpy as np
from scipy.linalg import eigh
from traser.representations.moments import weighted_batch_moments


def build(embeddings, fit_weights, output):
    x = np.load(embeddings, mmap_mode="r")
    w = np.load(fit_weights)
    assert x.ndim == 2 and x.shape[1] == 4096 and (w.shape == (len(x),))
    assert np.isfinite(w).all() and (w >= 0).all() and (w.sum() > 64)
    total = float(w.sum())
    sx = np.zeros(4096)
    xx = np.zeros((4096, 4096))
    for start in range(0, len(x), 4096):
        a = np.asarray(x[start : start + 4096])
        q = w[start : start + 4096]
        keep = q > 0
        assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=2e-05), (
            "4096D embeddings must already be L2 normalized"
        )
        if keep.any():
            first, second = weighted_batch_moments(a[keep], q[keep])
            sx += first
            xx += second
    mean = sx / total
    cov = xx / total - np.outer(mean, mean)
    values, vectors = eigh(cov, subset_by_index=[4096 - 64, 4095])
    basis = vectors[:, ::-1]
    for j in range(64):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    scale = np.sqrt(values.sum())
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "model.npz",
        mean=mean,
        components=basis,
        eigenvalues=values[::-1],
        rms=np.array(scale),
    )
    coords = np.lib.format.open_memmap(
        output / "coordinates.npy", mode="w+", dtype="float32", shape=(len(x), 64)
    )
    for start in range(0, len(x), 4096):
        coords[start : start + 4096] = (x[start : start + 4096] - mean) @ basis / scale
    coords.flush()
