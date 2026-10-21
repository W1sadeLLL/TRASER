"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from __future__ import annotations
import time
import numpy as np
import torch


def directions(seed, blocks=4, dimension=64):
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(blocks):
        q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        q *= np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
        bases.append(q)
    return np.concatenate(bases, axis=1)


class SlicedDistance:
    def __init__(self, particle_count, seed, blocks=4, axes=False):
        self.n = particle_count
        theta = np.eye(64) if axes else directions(seed, blocks)
        self.theta64 = torch.as_tensor(theta, device="cuda", dtype=torch.float64)
        self.theta = self.theta64.float()
        self.scale = 1 if axes else 64
        self.cache = {}

    @torch.no_grad()
    def prepare(self, observed, key):
        if key in self.cache:
            return self.cache[key]
        torch.cuda.synchronize()
        started = time.perf_counter()
        target = torch.as_tensor(observed, device="cuda", dtype=torch.float64)
        m = len(target)
        bins = torch.arange(self.n + 1, device="cuda", dtype=torch.int64)
        numer = bins * m
        index = torch.div(numer, self.n, rounding_mode="floor")
        fraction = (numer % self.n).double() / self.n
        means, variances = ([], [])
        for start in range(0, self.theta64.shape[1], 32):
            projected = (target @ self.theta64[:, start : start + 32]).T.contiguous()
            values = projected.sort(dim=1).values
            prefix = torch.cat((values.new_zeros((len(values), 1)), values.cumsum(1)), 1)
            integrals = (
                prefix[:, index] + fraction[None, :] * values[:, index.clamp_max(m - 1)]
            ) / m
            qbar = (integrals[:, 1:] - integrals[:, :-1]) * self.n
            variance = values.square().mean(1) - qbar.square().mean(1)
            if bool((variance < -1e-09).any()):
                raise RuntimeError("Negative within-bin target quantile variance")
            means.append(qbar.T.contiguous())
            variances.append(variance.clamp_min(0))
        q = torch.cat(means, 1)
        variance = torch.cat(variances)
        center = target.mean(0)
        rms = (target - center).square().sum(1).mean().sqrt()
        torch.cuda.synchronize()
        result = {
            "q64": q,
            "q": q.float(),
            "variance64": variance,
            "variance": variance.float(),
            "count": m,
            "center": center,
            "rms": float(rms),
            "prepare_seconds": time.perf_counter() - started,
        }
        self.cache[key] = result
        return result

    def squared(self, prediction, target, precise=False):
        theta = self.theta64 if precise else self.theta
        x = prediction.to(theta.dtype)
        values = (x @ theta).sort(dim=0).values
        q = target["q64"] if precise else target["q"]
        variance = target["variance64"] if precise else target["variance"]
        return (values - q).square().mean(0) + variance

    def loss(self, prediction, observed, key, records):
        target = self.prepare(observed, key)
        torch.cuda.synchronize()
        started = time.perf_counter()
        squared = self.scale * self.squared(prediction, target).mean()
        value = (squared + 1e-12).sqrt() - 1e-06
        if not bool(torch.isfinite(value)):
            raise RuntimeError("Nonfinite sliced forecast loss")
        records.append(
            {
                "scaled_sliced_w2": float(value.detach()),
                "target_occurrences": target["count"],
                "directions": self.theta.shape[1],
                "full_target_mass": True,
                "elapsed_seconds": time.perf_counter() - started,
                "solver_backend": "CUDA sorting and exact 1D quantile integrals",
            }
        )
        return value

    @torch.no_grad()
    def score(self, prediction, observed, key):
        target = self.prepare(observed, key)
        torch.cuda.synchronize()
        started = time.perf_counter()
        squared = self.scale * self.squared(prediction, target, precise=True).mean()
        delta = prediction.double().mean(0) - target["center"]
        center2 = self.scale * (delta @ self.theta64).square().mean()
        shape2 = squared - center2
        if not bool(torch.isfinite(squared)) or float(shape2) < -1e-08:
            raise RuntimeError("Invalid sliced distance or centered decomposition")
        torch.cuda.synchronize()
        return {
            "scaled_sliced_w2": float(squared.clamp_min(0).sqrt()),
            "scaled_sliced_w2_squared": float(squared),
            "centered_scaled_sliced_w2": float(shape2.clamp_min(0).sqrt()),
            "mean_displacement_l2": float(delta.norm()),
            "population_mean_MSE": float(delta.square().mean()),
            "target_occurrences": target["count"],
            "full_target_mass": True,
            "directions": self.theta.shape[1],
            "solver_backend": "CUDA float64 exact projected quantile integral",
            "elapsed_seconds": time.perf_counter() - started,
        }


@torch.no_grad()
def bcm_step(x, alpha, radius, row_block=1024):
    if alpha == 0:
        return x
    norm = x.square().sum(1)
    result = torch.empty_like(x)
    for start in range(0, len(x), row_block):
        left = x[start : start + row_block]
        squared = norm[start : start + row_block, None] + norm[None, :] - 2 * (left @ x.T)
        tolerance = 2e-05 * (norm[start : start + row_block, None] + norm[None, :] + radius**2 + 1)
        near = (squared - radius**2).abs() <= tolerance
        rows, cols = torch.where(near)
        if len(rows):
            squared[rows, cols] = (left[rows] - x[cols]).square().sum(1)
        active = (squared <= radius**2).to(x.dtype)
        local = torch.arange(len(left), device=x.device)
        active[local, local + start] = 1
        result[start : start + len(left)] = left + alpha * (
            active @ x / active.sum(1, keepdim=True) - left
        )
    return result


def mmd_update(prior, target, strength, bandwidth, cfg):
    """Same complete Gaussian MMD objective, bounded iterations, resident data."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    anchor = prior.detach()
    state = torch.nn.Parameter(anchor.clone())
    observed, mass = (target["points"], target["mass"])
    optimizer = torch.optim.Adam([state], lr=cfg["learning_rate"])
    best, best_value, best_step, curve = (anchor.clone(), float("inf"), 0, [])
    n = len(state)
    for iteration in range(cfg["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        objective = state.new_zeros(())
        for start in range(0, n, cfg["row_block"]):
            term = (
                torch.exp(
                    -torch.cdist(state[start : start + cfg["row_block"]], state).square()
                    / (2 * bandwidth**2)
                ).sum()
                / n**2
            )
            objective += term.detach()
            if iteration < cfg["steps"]:
                term.backward()
        for start in range(0, len(observed), cfg["target_block"]):
            end = min(start + cfg["target_block"], len(observed))
            term = (
                -2
                * (
                    torch.exp(
                        -torch.cdist(state, observed[start:end]).square() / (2 * bandwidth**2)
                    )
                    * mass[start:end]
                ).sum()
                / n
            )
            objective += term.detach()
            if iteration < cfg["steps"]:
                term.backward()
        tether = strength * (state - anchor).square().sum(1).mean()
        objective += tether.detach()
        value = float(objective)
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite full-mass MMD objective")
        if value < best_value:
            best, best_value, best_step = (state.detach().clone(), value, iteration)
        curve.append({"step": iteration, "offset_objective": value})
        if iteration < cfg["steps"]:
            tether.backward()
            if not bool(torch.isfinite(state.grad).all()):
                raise RuntimeError("Nonfinite MMD gradient")
            optimizer.step()
    torch.cuda.synchronize()
    return (
        best,
        {
            "method": "full_mass_Gaussian_MMD",
            "strength": strength,
            "steps": cfg["steps"],
            "selected_step": best_step,
            "curve": curve,
            "target_occurrences": target["count"],
            "distinct_coordinates": len(observed),
            "full_target_mass": True,
            "wall_seconds": time.perf_counter() - started,
        },
    )
