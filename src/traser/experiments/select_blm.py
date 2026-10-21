"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import time, itertools, hashlib, json
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
from traser.experiments.chains import chain


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def selection(exp, cfg, deadline):
    path = exp.out / "selection.json"
    if path.exists():
        return read(path)
    grid = read(ROOT / "configs/search/blm_mechanical_grid.json")
    q25, q50, q75 = exp.features["radii"]
    radii = [0.5 * q25, q25, q50, q75, 1.5 * q75]
    candidates = {}

    def propose(a, r, s):
        a = float(a)
        v = {"alpha": a, "radius": float(r if a else q50), "sigma": float(s)}
        candidates[key(v)] = v
        return v

    for a, r, s in itertools.product(grid["alpha_grid"], radii, grid["sigma_grid"]):
        propose(a, r, s)
    old = read(ROOT / "assets" / exp.dataset / "mechanical_anchor.json")["bcm"]
    fraction = exp.window_hours / 12
    anchor = propose(
        1 - (1 - old["alpha"]) ** fraction, old["radius"], old["sigma"] * np.sqrt(fraction)
    )
    records = {}
    first = True

    def mechanical(v, stage):
        nonlocal first
        k = key(v)
        file = exp.out / "mechanical_candidates" / (k + ".json")
        if file.exists():
            r = read(file)
        else:
            tick = time.time()
            rr = []
            for seed in cfg["seeds"]:
                rr.extend(chain(exp, "development", "H0", seed, v, None)["rows"])
            r = {
                "bcm": v,
                "score": float(np.mean([a["scaled_sliced_w2"] for a in rr])),
                "rows": rr,
                "seconds": time.time() - tick,
                "stage": stage,
                "test_used": False,
            }
            write(file, r)
            if first:
                estimate = 1.2 * r["seconds"] * 400 + 6 * 3600
                write(
                    exp.out / "selection_estimate.json",
                    {
                        "first_full_mechanical_unit_seconds": r["seconds"],
                        "upper_allowance_seconds": estimate,
                        "remaining_seconds": deadline - time.time(),
                    },
                )
                if time.time() + estimate > deadline:
                    raise TimeoutError("Complete selection cannot fit remaining original budget")
                first = False
        records[k] = r
        exp.progress(
            "MECHANICAL_SELECTION", completed=len(records), stage_name=stage, score=r["score"]
        )

    mechanical(anchor, "duration_scaled_previous_optimum")
    for v in candidates.values():
        mechanical(v, "coarse")
    rank = lambda r: (r["score"], r["bcm"]["alpha"], r["bcm"]["sigma"], r["bcm"]["radius"])
    for iteration in range(2):
        v = min(records.values(), key=rank)["bcm"]
        aa = (
            [0, 0.0025, 0.005]
            if v["alpha"] == 0
            else [min(1, v["alpha"] * f) for f in (0.5, 1, 1.5)]
        )
        for a, r, s in itertools.product(
            aa,
            [v["radius"] * f for f in (0.75, 1, 1.25)],
            [min(0.2, max(1e-05, v["sigma"] * f)) for f in (0.5, 1, 2)],
        ):
            mechanical(propose(a, r, s), "refinement_" + str(iteration + 1))
    best = min(records.values(), key=rank)
    bcm = best["bcm"]
    write(
        exp.out / "bcm_selection.json",
        {"bcm": bcm, "score": best["score"], "candidates": len(records), "test_used": False},
    )
    updates = {}

    def update(spec, stage):
        k = key(spec)
        file = exp.out / "assimilation_candidates" / (k + ".json")
        if file.exists():
            r = read(file)
        else:
            tick = time.time()
            rr = []
            uu = []
            for seed in cfg["seeds"]:
                result = chain(exp, "development", "HA", seed, bcm, spec)
                rr.extend(result["rows"])
                uu.extend(({"seed": seed, **q} for q in result["updates"]))
            r = {
                "spec": spec,
                "score": float(np.mean([a["scaled_sliced_w2"] for a in rr])),
                "rows": rr,
                "updates": uu,
                "seconds": time.time() - tick,
                "stage": stage,
                "test_used": False,
            }
            write(file, r)
        updates[k] = r
        exp.progress(
            "ASSIMILATION_SELECTION",
            completed=len(updates),
            family=spec["family"],
            stage_name=stage,
            score=r["score"],
        )

    for solver in cfg["mmd_solvers"]:
        for strength in cfg["mmd_strengths"]:
            update(
                {
                    "family": "MMD",
                    "strength": strength,
                    **solver,
                    "count_power": 0,
                    "innovation_response": 0,
                },
                "fixed",
            )
    for strength in cfg["transport_strengths"]:
        update(
            {
                "family": "sliced_transport",
                "strength": strength,
                "steps": cfg["transport_steps"],
                "count_power": 0,
                "innovation_response": 0,
            },
            "fixed",
        )
    update_rank = lambda r: (
        r["score"],
        r["spec"]["count_power"] + r["spec"]["innovation_response"],
        r["spec"]["steps"],
        r["spec"]["family"],
        -r["spec"]["strength"],
    )
    fixed = min(updates.values(), key=update_rank)
    for power, response, factor in itertools.product(
        cfg["count_powers"], cfg["innovation_responses"], cfg["adaptive_strength_factors"]
    ):
        spec = {
            **fixed["spec"],
            "strength": fixed["spec"]["strength"] * factor,
            "count_power": power,
            "innovation_response": response,
        }
        update(spec, "adaptive")
    chosen = min(updates.values(), key=update_rank)
    result = {
        "bcm": bcm,
        "pure_bcm_development_score": best["score"],
        "mechanical_candidates": len(records),
        "fixed": fixed,
        "selected": chosen,
        "assimilation_candidates": len(updates),
        "selected_utc": stamp(),
        "test_used": False,
        "window_hours": exp.window_hours,
    }
    write(path, result)
    return result
