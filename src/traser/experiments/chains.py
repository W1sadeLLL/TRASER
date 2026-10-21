"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import os, time, json, hashlib, itertools, math
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
import traser.dynamics.base as parent

CONFIG = ROOT / "configs/experiments/blm.json"
OUT = ROOT / "outputs/replay"


def chain(exp, phase, group, seed, bcm, spec, folder=None, diagnostics=False):
    import torch

    if folder is not None and (folder / "complete.json").exists():
        return read(folder / "complete.json")
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        parent.seal(
            folder / "identity.json",
            {
                "experiment": exp.analysis_identity,
                "phase": phase,
                "group": group,
                "seed": seed,
                "bcm": bcm,
                "update": spec,
                "diagnostics": diagnostics,
            },
        )
    windows = exp.phases[phase]
    x = exp.initial(phase, seed)
    rows = []
    reconstruction = []
    updates = []
    start = time.time()

    def recon(value, t, kind):
        path = folder / f"window_{t}_{kind}.npy"
        parent.atomic_array(value.cpu().numpy(), path)
        reconstruction.append(
            {
                "dataset": exp.dataset,
                "window_hours": exp.window_hours,
                "particles": exp.budget,
                "group": group,
                "phase": phase,
                "seed": seed,
                "window": t,
                "kind": kind,
                "initialization_only": kind == "initial",
                "duration_hours": exp.windows[t]["duration_hours"],
                "state_file": str(path.relative_to(OUT)),
                **exp.semantic(value, phase, t),
            }
        )

    if diagnostics:
        recon(x, windows[0], "initial")
    for t in windows[:-1]:
        target = t + 1
        fraction = exp.windows[target]["duration_hours"] / exp.window_hours
        step_bcm = {
            **bcm,
            "alpha": 1 - (1 - bcm["alpha"]) ** fraction,
            "sigma": bcm["sigma"] * np.sqrt(fraction),
        }
        with torch.no_grad():
            prior = exp.transition(
                None, x, None, step_bcm, exp.innovation(phase, seed, t, 1), False
            )[0]
        full = exp.windows[t]["full_duration"] and exp.windows[target]["full_duration"]
        if full and exp.windows[target]["eligible_occurrences"]:
            scored = exp.score(prior, phase, target)
            row = {
                "dataset": exp.dataset,
                "window_hours": exp.window_hours,
                "forecast_hours": exp.window_hours,
                "horizon": 1,
                "particles": exp.budget,
                "phase": phase,
                "group": group,
                "seed": seed,
                "origin": t,
                "target": target,
                "current_observation_count": exp.windows[t]["eligible_occurrences"],
                **scored,
            }
            if diagnostics and phase == "test":
                path = folder / f"origin_{t}_h_1.npy"
                parent.atomic_array(prior.cpu().numpy(), path)
                m = exp.metrics(prior.cpu().numpy(), target, scored)
                density_keys = {
                    "symmetric_KL",
                    "sqrt_JS_arithmetic",
                    "sqrt_JS_geometric_unnormalized",
                    "density_L1",
                }
                for name, value in m["metrics"].items():
                    if name not in density_keys and (value is None or not np.isfinite(value)):
                        raise RuntimeError("Invalid non-density forecast metric: " + name)
                    if value is not None and (not np.isfinite(value)):
                        raise RuntimeError("Nonfinite forecast metric: " + name)
                allowed = {
                    "KDE undefined with fewer than 2 occurrences",
                    "Degenerate marginal KDE bandwidth",
                }
                if any((v["reason"] not in allowed for v in m["density_failures"])):
                    raise RuntimeError("Numerical density failure: " + str(m["density_failures"]))
                row["density_failures"] = m["density_failures"]
                row["density_status"] = (
                    "UNDEFINED_SPARSE_OR_DEGENERATE_TARGET" if m["density_failures"] else "DEFINED"
                )
                row["reporting_repair"] = "hssc_adaptive_v3_sparse_v1"
                row.update(m["metrics"])
                row["semantic"] = exp.semantic(prior, phase, target)
                row["prediction"] = str(path.relative_to(OUT))
            rows.append(row)
        if diagnostics:
            recon(prior, target, "prior")
        if group != "H0" and (diagnostics or target != windows[-1]):
            x, diag = exp.adaptive_update(prior, phase, target, spec)
            updates.append({"origin": target, **diag})
        else:
            x = prior
        if diagnostics:
            recon(x, target, "analysis")
    result = {
        "state": "COMPLETE",
        "rows": rows,
        "reconstruction": reconstruction,
        "updates": updates,
        "seconds": time.time() - start,
        "score": float(np.mean([r["scaled_sliced_w2"] for r in rows])),
        "full_calendar_windows": len(windows),
        "scored_origins": len(rows),
        "empty_windows": [t for t in windows if exp.windows[t]["eligible_occurrences"] == 0],
        "partial_windows": [t for t in windows if not exp.windows[t]["full_duration"]],
    }
    if folder is not None:
        write(folder / "complete.json", result)
    return result


def paired(rows, hours, reference="H0"):
    cfg = read(CONFIG)
    result = []
    metrics = [
        "scaled_sliced_w2",
        "projection_sensitivity_scaled_sliced_w2",
        "marginal_W2_mean",
        "population_mean_MSE",
        "mean_displacement_l2",
        "centered_scaled_sliced_w2",
        "symmetric_KL",
        "sqrt_JS_arithmetic",
        "sqrt_JS_geometric_unnormalized",
        "density_L1",
        "marginal_W1_mean",
        "marginal_KS_mean",
        "covariance_Frobenius",
    ]
    for d in cfg["datasets"]:
        ar = [r for r in rows if r["dataset"] == d and r["window_hours"] == hours]
        for metric in metrics:
            all_origins = sorted({r["origin"] for r in ar})
            origins = [
                t
                for t in all_origins
                if len([r for r in ar if r["origin"] == t]) == 9
                and all(
                    (
                        r.get(metric) is not None and np.isfinite(r[metric])
                        for r in ar
                        if r["origin"] == t
                    )
                )
            ]
            if not origins:
                result.append(
                    {
                        "dataset": d,
                        "window_hours": hours,
                        "reference": reference,
                        "metric": metric,
                        "HA_mean": None,
                        reference + "_mean": None,
                        "mean_difference": None,
                        "relative_reduction_percent": None,
                        "origins": {},
                        "improved_origins": 0,
                        "origin_count": 0,
                        "total_origin_count": len(all_origins),
                        "missing_origins": all_origins,
                        "blocks": 0,
                        "interval_95": None,
                        "leave_one_out": {},
                    }
                )
                continue
            delta = []
            aa = []
            bb = []
            for t in origins:
                a = np.mean([r[metric] for r in ar if r["origin"] == t and r["group"] == "HA"])
                b = np.mean([r[metric] for r in ar if r["origin"] == t and r["group"] == reference])
                aa.append(a)
                bb.append(b)
                delta.append(a - b)
            groups = {}
            for t, v in zip(origins, delta):
                groups.setdefault((t - all_origins[0]) // max(1, 48 // hours), []).append(v)
            blocks = list(groups.values())
            interval = None
            if len(blocks) >= 2:
                rng = np.random.default_rng(2026101903)
                boot = [
                    np.mean(
                        np.concatenate(
                            [blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]
                        )
                    )
                    for _ in range(10000)
                ]
                interval = np.quantile(boot, [0.025, 0.975]).tolist()
            result.append(
                {
                    "dataset": d,
                    "window_hours": hours,
                    "reference": reference,
                    "metric": metric,
                    "HA_mean": float(np.mean(aa)),
                    reference + "_mean": float(np.mean(bb)),
                    "mean_difference": float(np.mean(delta)),
                    "relative_reduction_percent": float(
                        100 * (np.mean(bb) - np.mean(aa)) / np.mean(bb)
                    ),
                    "origins": dict(zip(map(str, origins), map(float, delta))),
                    "improved_origins": sum((v < -1e-12 for v in delta)),
                    "origin_count": len(origins),
                    "total_origin_count": len(all_origins),
                    "missing_origins": [t for t in all_origins if t not in origins],
                    "blocks": len(blocks),
                    "block_duration_hours": 48,
                    "interval_95": interval,
                    "leave_one_out": {
                        str(t): float(np.mean([v for tt, v in zip(origins, delta) if tt != t]))
                        if len(origins) > 1
                        else None
                        for t in origins
                    },
                }
            )
    return result
