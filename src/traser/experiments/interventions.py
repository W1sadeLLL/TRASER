"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp


def weighted_hk(x, alpha, radius, weights, block):
    """Same hard radius and self edge as the parent operator."""
    import torch

    norms = x.square().sum(1)
    result = torch.empty_like(x)
    for start in range(0, len(x), block):
        left = x[start : start + block]
        squared = norms[start : start + block, None] + norms[None, :] - 2 * (left @ x.T)
        tolerance = 2e-05 * (norms[start : start + block, None] + norms[None, :] + radius**2 + 1)
        near = (squared - radius**2).abs() <= tolerance
        rows, cols = torch.where(near)
        if len(rows):
            squared[rows, cols] = (left[rows] - x[cols]).square().sum(1)
        active = (squared <= radius**2).to(x.dtype) * weights[None, :]
        local = torch.arange(len(left), device=x.device)
        active[local, local + start] = 1
        result[start : start + len(left)] = left + alpha * (
            active @ x / active.sum(1, keepdim=True) - left
        )
    return result


def matched_indices(x, scores, fraction):
    n = len(x)
    count = round(fraction * n)
    target = np.lexsort((np.arange(n), -scores))[:count]
    pool = np.setdiff1d(np.arange(n), target)
    descriptors = np.column_stack(
        (np.linalg.norm(x, axis=1), np.linalg.norm(x - x.mean(0), axis=1))
    )
    scale = np.maximum(descriptors.std(0), 1e-08)
    z = descriptors / scale
    available = np.ones(len(pool), dtype=bool)
    matched = []
    dist = []
    for i in target:
        candidates = np.flatnonzero(available)
        ds = ((z[pool[candidates]] - z[i]) ** 2).sum(1)
        k = candidates[int(np.argmin(ds))]
        matched.append(int(pool[k]))
        dist.append(float(np.sqrt(ds.min())))
        available[k] = False
    controls = np.array(matched)
    return (
        target,
        controls,
        {
            "count": count,
            "descriptor_mean_standardized_difference": (
                (descriptors[target].mean(0) - descriptors[controls].mean(0)) / scale
            ).tolist(),
            "mean_match_distance": float(np.mean(dist)),
            "target_topic_mean": float(scores[target].mean()),
            "control_topic_mean": float(scores[controls].mean()),
        },
    )


def outcome(exp, x, baseline):
    import torch

    v, clip = exp.mapped(x)
    b, bc = exp.mapped(baseline)
    delta = v.mean(0) - b.mean(0)
    theta = exp.distance.theta64
    latent = (
        (x.double() @ theta).sort(0).values - (baseline.double() @ theta).sort(0).values
    ).square().mean() * 64
    rms = (x.double() - x.double().mean(0)).square().sum(1).mean().sqrt()
    return {
        "scores": v.mean(0).tolist(),
        "baseline_scores": b.mean(0).tolist(),
        "delta_scores": delta.tolist(),
        "semantic_W1_to_baseline": (v.sort(0).values - b.sort(0).values).abs().mean(0).tolist(),
        "clipped_fraction": clip.tolist(),
        "baseline_clipped_fraction": bc.tolist(),
        "rms": float(rms),
        "scaled_sliced_W2_to_baseline": float(latent.sqrt()),
    }


def simulate(exp, initial, bcm, seed, origin, name, weights=None):
    import torch
    from traser.dynamics.operators import bcm_step

    result = {}
    x = initial.clone()
    for k in range(1, 2):
        exp.guard()
        with torch.no_grad():
            if weights is None:
                y = bcm_step(x, bcm["alpha"], bcm["radius"], exp.cfg["resources"]["bcm_row_block"])
            else:
                y = weighted_hk(
                    x, bcm["alpha"], bcm["radius"], weights, exp.cfg["resources"]["bcm_row_block"]
                )
            x = y + bcm["sigma"] * exp.innovation("test", seed, origin, k)
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Nonfinite intervention state")
        if k == 1:
            result[k] = x.detach().clone()
    return result
