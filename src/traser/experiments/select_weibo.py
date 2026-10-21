"""Current paper implementation; provenance is recorded in provenance/module_map.json."""

from pathlib import Path
import time, itertools, hashlib, json
import numpy as np
from traser.io import ROOT, read, write, sha, stamp
import traser.assimilation.experiment as core
import traser.experiments.chains as reporting


def select(exp, cfg):
    path = exp.out / "selection.json"
    if path.exists():
        return read(path)
    q25, q50, q75 = exp.features["radii"]
    records, proposals = ({}, {})

    def propose(a, r, s):
        value = {"alpha": float(a), "radius": float(r if a else q50), "sigma": float(s)}
        proposals[core.key(value)] = value
        return value

    def evaluate(value, stage):
        key = core.key(value)
        file = exp.out / "mechanical_candidates" / (key + ".json")
        if file.exists():
            record = read(file)
        else:
            started = time.time()
            rows = []
            for seed in cfg["seeds"]:
                rows.extend(reporting.chain(exp, "development", "H0", seed, value, None)["rows"])
            record = {
                "bcm": value,
                "score": float(np.mean([r["scaled_sliced_w2"] for r in rows])),
                "rows": rows,
                "stage": stage,
                "test_used": False,
                "seconds": time.time() - started,
            }
            write(file, record)
        records[key] = record
        exp.progress(
            "MECHANICAL_SELECTION",
            completed=len(records),
            candidate_stage=stage,
            score=record["score"],
        )

    for a, r, s in itertools.product(
        cfg["alpha_grid"], [0.5 * q25, q25, q50, q75, 1.5 * q75], cfg["sigma_grid"]
    ):
        propose(a, r, s)
    for value in list(proposals.values()):
        evaluate(value, "coarse")
    rank = lambda r: (r["score"], r["bcm"]["alpha"], r["bcm"]["sigma"], r["bcm"]["radius"])
    for iteration in range(cfg["refinement_rounds"]):
        best = min(records.values(), key=rank)["bcm"]
        alphas = (
            [0, 0.0025, 0.005]
            if best["alpha"] == 0
            else [min(1, best["alpha"] * f) for f in (0.5, 1, 1.5)]
        )
        for a, r, s in itertools.product(
            alphas,
            [best["radius"] * f for f in (0.75, 1, 1.25)],
            [min(0.2, max(1e-05, best["sigma"] * f)) for f in (0.5, 1, 2)],
        ):
            evaluate(propose(a, r, s), f"refinement_{iteration + 1}")
    mechanical = min(records.values(), key=rank)
    bcm = mechanical["bcm"]
    write(
        exp.out / "bcm_selection.json",
        {"bcm": bcm, "score": mechanical["score"], "candidates": len(records), "test_used": False},
    )
    updates = {}

    def update(spec, stage):
        key = core.key(spec)
        file = exp.out / "assimilation_candidates" / (key + ".json")
        if file.exists():
            record = read(file)
        else:
            started = time.time()
            rows, diagnostics = ([], [])
            for seed in cfg["seeds"]:
                result = reporting.chain(exp, "development", "HA", seed, bcm, spec)
                rows.extend(result["rows"])
                diagnostics.extend(({"seed": seed, **d} for d in result["updates"]))
            record = {
                "spec": spec,
                "score": float(np.mean([r["scaled_sliced_w2"] for r in rows])),
                "rows": rows,
                "updates": diagnostics,
                "stage": stage,
                "test_used": False,
                "seconds": time.time() - started,
            }
            write(file, record)
        updates[key] = record
        exp.progress(
            "ASSIMILATION_SELECTION",
            completed=len(updates),
            family=spec["family"],
            candidate_stage=stage,
            score=record["score"],
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
    rank_update = lambda r: (
        r["score"],
        r["spec"]["count_power"] + r["spec"]["innovation_response"],
        r["spec"]["steps"],
        r["spec"]["family"],
        -r["spec"]["strength"],
    )
    fixed = min(updates.values(), key=rank_update)
    for power, response, factor in itertools.product(
        cfg["count_powers"], cfg["innovation_responses"], cfg["adaptive_strength_factors"]
    ):
        update(
            {
                **fixed["spec"],
                "strength": fixed["spec"]["strength"] * factor,
                "count_power": power,
                "innovation_response": response,
            },
            "adaptive",
        )
    chosen = min(updates.values(), key=rank_update)
    if chosen["score"] >= fixed["score"]:
        chosen = fixed
    result = {
        "bcm": bcm,
        "pure_bcm_development_score": mechanical["score"],
        "mechanical_candidates": len(records),
        "fixed": fixed,
        "selected": chosen,
        "assimilation_candidates": len(updates),
        "test_used": False,
        "selected_utc": stamp(),
        "window_hours": exp.window_hours,
    }
    write(path, result)
    return result
