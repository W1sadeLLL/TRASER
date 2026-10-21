"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp


def density(x, y):
    """Same occurrence-weighted KDE and quadrature rules; GPU float64 chunks."""
    import torch
    from scipy.special import ndtr

    if min(len(x), len(y)) < 2:
        raise ValueError("KDE undefined with fewer than 2 occurrences")
    hx = (3 * len(x) / 4) ** (-0.2) * np.std(x, ddof=1)
    hy = (3 * len(y) / 4) ** (-0.2) * np.std(y, ddof=1)
    if min(hx, hy) <= 1e-12:
        raise ValueError("Degenerate marginal KDE bandwidth")
    lo = min(x.min(), y.min()) - 8 * max(hx, hy)
    hi = max(x.max(), y.max()) + 8 * max(hx, hy)
    points = max(257, math.ceil((hi - lo) / (min(hx, hy) / 8)) + 1)
    grid = np.linspace(lo, hi, points)
    if points * 8 * 8 > 1000000000:
        raise ValueError("Full density grid exceeds declared 1GB vector storage; not coarsened")

    def evaluate(values, h):
        output = np.empty(points)
        values = torch.as_tensor(values, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            for a in range(0, points, 128):
                g = torch.as_tensor(grid[a : a + 128], dtype=torch.float64, device="cuda")
                logs = None
                for b in range(0, len(values), 8192):
                    part = torch.logsumexp(
                        -0.5 * ((g[:, None] - values[b : b + 8192][None, :]) / h).square(), dim=1
                    )
                    logs = part if logs is None else torch.logaddexp(logs, part)
                output[a : a + 128] = (
                    (logs - math.log(len(values) * h * math.sqrt(2 * math.pi))).cpu().numpy()
                )
        return output

    lx = evaluate(x, hx)
    ly = evaluate(y, hy)
    px = np.exp(lx)
    py = np.exp(ly)
    ax = float(np.mean(ndtr((hi - x) / hx) - ndtr((lo - x) / hx)))
    ay = float(np.mean(ndtr((hi - y) / hy) - ndtr((lo - y) / hy)))
    nx = float(np.trapezoid(px, grid))
    ny = float(np.trapezoid(py, grid))
    if max(abs(ax - nx), abs(ay - ny)) > 0.001:
        raise ValueError("KDE quadrature does not resolve its analytic mass")
    skl = float(np.trapezoid(px * (lx - ly) + py * (ly - lx), grid))
    mix = np.logaddexp(lx, ly) - math.log(2.0)
    js = float(0.5 * np.trapezoid(px * (lx - mix) + py * (ly - mix), grid))
    if min(skl, js) < -1e-08 or not np.isfinite([skl, js]).all():
        raise ValueError("Invalid density integral")
    return {
        "symmetric_KL": max(0.0, skl),
        "sqrt_JS_arithmetic": math.sqrt(max(0.0, js)),
        "sqrt_JS_geometric_unnormalized": 0.5 * math.sqrt(max(0.0, skl)),
        "density_L1": float(np.trapezoid(abs(px - py), grid)),
        "grid_points": points,
        "bandwidth_forecast": float(hx),
        "bandwidth_observed": float(hy),
    }
