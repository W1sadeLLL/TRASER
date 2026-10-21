"""Current complete conditional design; all seeds, topics, fractions, paired noise."""

import numpy as np
from traser.experiments.replay import Experiment, requirements
from traser.experiments.runtime import exclusive_experiment
from traser.io import ROOT, read, write
from traser.experiments.interventions import simulate, matched_indices, outcome


@exclusive_experiment
def run(dataset, asset_root=None):
    import torch

    root = (asset_root or ROOT / "assets").resolve()
    cfg, paths = requirements(dataset, root)
    missing = [str(p.relative_to(root)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing replay assets: " + ", ".join(missing))
    cfg_s = read(ROOT / "configs/experiments/scenarios.json")
    for hours in cfg["window_hours"]:
        exp = Experiment(dataset, hours, root)
        origin = exp.phases["test"][-1]
        bcm = read(root / dataset / f"w{hours}/selection.json")["bcm"]
        prototypes = dict(np.load(root / dataset / "mapping/prototypes.npz"))
        rows = []
        pairs = []
        for seed in cfg["seeds"]:
            source = exp.out / f"test/HA/seed_{seed}/window_{origin}_analysis.npy"
            if not source.exists():
                raise FileNotFoundError("Run full frozen replay first: " + str(source))
            x = torch.as_tensor(np.load(source), device="cuda")
            xn = x.cpu().numpy()
            scores = exp.mapped(x)[0].cpu().numpy()
            baseline = simulate(exp, x, bcm, seed, origin, "baseline")[1]
            for j, topic in enumerate(cfg_s["topics"]):
                for fraction in cfg_s["fractions"]:
                    target, controls, matching = matched_indices(xn, scores[:, j], fraction)
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                20261019,
                                0 if dataset == "blm_x" else 1,
                                seed,
                                j,
                                int(fraction * 1000),
                            ]
                        )
                    )
                    positions = rng.choice(len(x), round(fraction * len(x)), replace=False)
                    arms = {}
                    for condition in ["expression_injection", "matched_expression"]:
                        initial = x.clone()
                        p = prototypes[
                            topic + ("_sham" if condition == "matched_expression" else "")
                        ]
                        initial[positions] = torch.as_tensor(
                            p[np.arange(len(positions)) % len(p)], device="cuda", dtype=x.dtype
                        )
                        arms[condition] = simulate(exp, initial, bcm, seed, origin, condition)[1]
                    for condition, multiplier in cfg_s["source_multipliers"].items():
                        for suffix, indices in [("", target), ("_matched", controls)]:
                            weights = torch.ones(len(x), device="cuda", dtype=x.dtype)
                            weights[indices] = multiplier
                            arms[condition + suffix] = simulate(
                                exp, x, bcm, seed, origin, condition + suffix, weights
                            )[1]
                    common = {
                        "dataset": dataset,
                        "window_hours": hours,
                        "seed": seed,
                        "origin": origin,
                        "topic": topic,
                        "fraction": fraction,
                        "horizon": 1,
                        "matching": matching,
                    }
                    for condition, value in arms.items():
                        rows.append(
                            {**common, "condition": condition, **outcome(exp, value, baseline)}
                        )
                    for condition, control in [("expression_injection", "matched_expression")] + [
                        (c, c + "_matched") for c in cfg_s["source_multipliers"]
                    ]:
                        pairs.append(
                            {
                                **common,
                                "condition": condition,
                                "control": control,
                                **outcome(exp, arms[condition], arms[control]),
                            }
                        )
        assert len(rows) == 360 and len(pairs) == 180
        folder = ROOT / f"outputs/scenarios/{dataset}/w{hours}"
        write(folder / "all_outcomes.json", rows)
        write(folder / "paired_outcomes.json", pairs)
        exp.clear_phase()
