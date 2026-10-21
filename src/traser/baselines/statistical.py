"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp

GRID = np.linspace(0.0, 1.0, 65)
EPS = 1e-06


def _to_unit(points: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    if (
        points.ndim != 2
        or points.shape[1] != 64
        or (not len(points))
        or (not np.isfinite(points).all())
    ):
        raise ValueError("Expected nonempty finite PCA64 cloud")
    return expit(8 * points)


def _from_unit(values: np.ndarray) -> np.ndarray:
    from scipy.special import logit

    return logit(np.clip(values, EPS, 1 - EPS)) / 8


def _quantile(points: np.ndarray) -> np.ndarray:
    return np.quantile(_to_unit(points), GRID, axis=0).T


def _project_quantiles(q: np.ndarray) -> np.ndarray:
    projected = np.empty_like(q)
    for r in range(q.shape[0]):
        means = []
        counts = []
        for value in q[r]:
            means.append(float(value))
            counts.append(1)
            while len(means) > 1 and means[-2] > means[-1]:
                count = counts[-2] + counts[-1]
                mean = (means[-2] * counts[-2] + means[-1] * counts[-1]) / count
                means[-2:] = [mean]
                counts[-2:] = [count]
        projected[r] = np.repeat(means, counts)
    return np.clip(projected, EPS, 1 - EPS)


def _reconstruct(origin: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    transformed = _to_unit(origin)
    n = len(origin)
    result = np.empty_like(transformed)
    for r in range(64):
        u = (rankdata(transformed[:, r], method="average") - 0.5) / n
        result[:, r] = np.interp(u, GRID, quantiles[r])
    return _from_unit(result)


def fit_war(training: dict[int, np.ndarray]) -> dict:
    if len(training) < 3:
        raise RuntimeError("WAR requires at least three observed training windows")
    q = {t: _quantile(cloud) for t, cloud in sorted(training.items())}
    barycenter = np.mean(list(q.values()), axis=0)
    lag_pairs = [(t, t + 1) for t in q if t + 1 in q]
    if not lag_pairs:
        raise RuntimeError("WAR has no consecutive observed training transitions")
    numerator = np.zeros(64)
    denominator = np.zeros(64)
    for t, next_t in lag_pairs:
        vt = q[t] - barycenter
        vn = q[next_t] - barycenter
        numerator += np.trapezoid(vt * vn, GRID, axis=1)
        denominator += np.trapezoid(vt * vt, GRID, axis=1)
    beta = np.divide(numerator, denominator, out=np.zeros(64), where=denominator > 0)
    return {
        "model": "WAR",
        "barycenter": barycenter,
        "beta": beta,
        "train_windows": tuple(sorted(training)),
        "consecutive_pairs": tuple(lag_pairs),
        "degenerate_axes": int(np.count_nonzero(denominator == 0)),
    }


def predict_war(model: dict, origin: np.ndarray, horizon: int) -> np.ndarray:
    if horizon not in (1, 2, 4):
        raise ValueError("Invalid horizon")
    quantiles = _quantile(origin)
    barycenter = model["barycenter"]
    for _ in range(horizon):
        quantiles = _project_quantiles(
            barycenter + model["beta"][:, None] * (quantiles - barycenter)
        )
    return _reconstruct(origin, quantiles)


def fit_wr(training: dict[int, np.ndarray], rank: int) -> dict:
    if rank not in (1, 2):
        raise ValueError("WR rank must be 1 or 2")
    times = sorted(training)
    q = {t: _quantile(training[t]) for t in times}
    average = np.mean(list(q.values()), axis=0)
    lag_pairs = [(t, t + 1) for t in times if t + 1 in q]
    if len(lag_pairs) < 2:
        raise RuntimeError("WR requires at least two observed consecutive training transitions")
    basis = np.empty((64, rank, len(GRID)))
    eigen = np.empty((64, rank))
    gamma = np.empty((64, rank, rank))
    weights = np.full(len(GRID), 1 / (len(GRID) - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5
    for r in range(64):
        centered = np.stack([q[t][r] - average[r] for t in times])
        _, singular, vt = np.linalg.svd(centered * np.sqrt(weights)[None, :], full_matrices=False)
        phi = vt[:rank] / np.sqrt(weights)[None, :]
        scores = centered @ (phi * weights).T
        eig = singular[:rank] ** 2 / len(times)
        lag = np.array([times.index(t) for t, _ in lag_pairs])
        cross = scores[lag].T @ scores[lag + 1] / len(lag_pairs)
        ratio = np.divide(cross, eig[:, None], out=np.zeros_like(cross), where=eig[:, None] > 1e-12)
        basis[r], eigen[r], gamma[r] = (phi, eig, ratio)
    return {
        "model": "WR",
        "rank": rank,
        "average": average,
        "basis": basis,
        "eigenvalues": eigen,
        "operator": gamma,
        "train_windows": tuple(times),
        "consecutive_pairs": tuple(lag_pairs),
    }


def predict_wr(model: dict, origin: np.ndarray, horizon: int) -> np.ndarray:
    if horizon not in (1, 2, 4):
        raise ValueError("Invalid horizon")
    q = _quantile(origin)
    for _ in range(horizon):
        next_q = np.empty_like(q)
        for r in range(64):
            centered = q[r] - model["average"][r]
            scores = np.trapezoid(centered[None, :] * model["basis"][r], GRID, axis=1)
            next_q[r] = model["average"][r] + scores @ model["operator"][r] @ model["basis"][r]
        q = _project_quantiles(next_q)
    return _reconstruct(origin, q)


def _lqd_transform(points: np.ndarray, bandwidth: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    if bandwidth not in (0.025, 0.05):
        raise ValueError("LQD bandwidth outside frozen grid")
    unit = _to_unit(points)
    bins = 128
    centers = (np.arange(bins) + 0.5) / bins
    result = np.empty((64, len(GRID)))
    for r in range(64):
        counts = np.histogram(unit[:, r], bins=bins, range=(0, 1))[0].astype(float)
        density = gaussian_filter1d(counts, sigma=bandwidth * bins, mode="reflect")
        density = density / max(density.sum(), EPS) * bins
        density = 0.999 * density + 0.001
        density /= density.mean()
        cdf = np.cumsum(density) / density.sum()
        cdf = np.r_[0.0, cdf]
        edges = np.linspace(0, 1, bins + 1)
        quantiles = np.interp(GRID, cdf, edges)
        q_density = 1 / np.interp(quantiles, centers, density, left=density[0], right=density[-1])
        log_q = np.log(q_density + EPS)
        result[r] = log_q - np.trapezoid(log_q, GRID)
    if not np.isfinite(result).all():
        raise RuntimeError("Invalid LQD transform")
    return result


def fit_lqd(training: dict[int, np.ndarray], bandwidth: float, rank: int, ridge: float) -> dict:
    if bandwidth not in (0.025, 0.05) or rank not in (1, 2) or ridge not in (0.1, 1.0, 10.0):
        raise ValueError("LQD parameters outside frozen grid")
    times = sorted(training)
    lag_pairs = [(t, t + 1) for t in times if t + 1 in training]
    if len(lag_pairs) < 2:
        raise RuntimeError("LQD requires two consecutive observed training transitions")
    g = {t: _lqd_transform(training[t], bandwidth) for t in times}
    average = np.mean(list(g.values()), axis=0)
    weights = np.full(len(GRID), 1 / (len(GRID) - 1))
    weights[0] *= 0.5
    weights[-1] *= 0.5
    basis = np.empty((64, rank, len(GRID)))
    matrix = np.empty((64, rank, rank))
    intercept = np.empty((64, rank))
    for r in range(64):
        centered = np.stack([g[t][r] - average[r] for t in times])
        _, _, vt = np.linalg.svd(centered * np.sqrt(weights)[None, :], full_matrices=False)
        phi = vt[:rank] / np.sqrt(weights)[None, :]
        scores = centered @ (phi * weights).T
        inputs = np.stack([scores[times.index(t)] for t, _ in lag_pairs])
        outputs = np.stack([scores[times.index(t + 1)] for t, _ in lag_pairs])
        xmean, ymean = (inputs.mean(0), outputs.mean(0))
        centered_x, centered_y = (inputs - xmean, outputs - ymean)
        transfer = np.linalg.solve(
            centered_x.T @ centered_x + ridge * np.eye(rank), centered_x.T @ centered_y
        )
        basis[r], matrix[r], intercept[r] = (phi, transfer, ymean - xmean @ transfer)
    if (
        not np.isfinite(basis).all()
        or not np.isfinite(matrix).all()
        or (not np.isfinite(intercept).all())
    ):
        raise RuntimeError("Nonfinite fitted LQD parameters")
    return {
        "model": "LQD",
        "bandwidth": bandwidth,
        "rank": rank,
        "ridge": ridge,
        "average": average,
        "basis": basis,
        "matrix": matrix,
        "intercept": intercept,
        "train_windows": tuple(times),
        "consecutive_pairs": tuple(lag_pairs),
        "inverse_quantile_grid": 4097,
        "density_bins": 128,
        "density_regularization": 0.001,
    }


def predict_lqd(model: dict, origin: np.ndarray, horizon: int) -> np.ndarray:
    if horizon not in (1, 2, 4):
        raise ValueError("Invalid horizon")
    incoming = _lqd_transform(origin, model["bandwidth"])
    quantiles = np.empty((64, len(GRID)))
    fine = np.linspace(0, 1, 4097)
    for r in range(64):
        scores = np.trapezoid(
            (incoming[r] - model["average"][r])[None, :] * model["basis"][r], GRID, axis=1
        )
        for _ in range(horizon):
            scores = scores @ model["matrix"][r] + model["intercept"][r]
        forecast_g = model["average"][r] + scores @ model["basis"][r]
        dense_g = np.interp(fine, GRID, forecast_g)
        dense_g -= dense_g.max()
        q = np.exp(dense_g)
        area = np.trapezoid(q, fine)
        if not np.isfinite(area) or area <= 0:
            raise RuntimeError("Invalid LQD inverse quantile density")
        q /= area
        q_function = np.r_[0.0, np.cumsum((q[1:] + q[:-1]) * np.diff(fine) / 2)]
        if not np.isfinite(q_function).all() or not np.isclose(q_function[-1], 1.0, atol=1e-08):
            raise RuntimeError("Invalid LQD inverse CDF")
        quantiles[r] = np.interp(GRID, fine, q_function)
    return _reconstruct(origin, quantiles)
