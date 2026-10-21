"""Portable constructor around unchanged current HK/assimilation operators.

Outputs are new replay results; they never overwrite frozen scientific evidence.
"""

from pathlib import Path
import json, numpy as np
from traser.io import ROOT, read, write, sha
from traser.assimilation.experiment import Experiment as NumericalExperiment
from traser.experiments.runtime import exclusive_experiment


class Experiment(NumericalExperiment):
    def __init__(self, dataset, hours, asset_root, particles=None):
        import torch, pyarrow.parquet as pq
        from traser.dynamics.operators import SlicedDistance

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Full-mass numerical replay requires CUDA PyTorch. CPU import checks are not scientific replay."
            )
        self.torch = torch
        self.dataset = dataset
        self.window_hours = hours
        self.adaptive_cfg = read(
            ROOT / f"configs/experiments/{('blm' if dataset == 'blm_x' else 'weibo')}.json"
        )
        self.cfg = read(ROOT / "configs/numerics/default.json")
        self.budget = particles or self.adaptive_cfg["particles"][dataset]
        budget_suffix = "" if particles is None else f"/N{particles}"
        self.out = ROOT / f"outputs/replay/{dataset}/w{hours}{budget_suffix}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.parent_out = self.out
        self.base_budget = self.budget
        folder = asset_root / dataset / f"w{hours}"
        self.calendar = read(folder / "calendar.json")
        self.windows = {w["window"]: w for w in self.calendar["windows"]}
        self.phases = {
            p: [t for t, w in self.windows.items() if w["phase"] == p]
            for p in ("train", "development", "test")
        }
        self.observation_index = pq.read_table(
            folder / "observations.parquet", columns=["coordinate_row", "window"]
        )
        column = pq.read_table(
            asset_root / dataset / "representation/coordinates.parquet", columns=["coordinates"]
        )["coordinates"].combine_chunks()
        self.all_coordinates = np.asarray(column.values.to_numpy()).reshape(-1, 64)
        self.distance = SlicedDistance(
            self.budget, self.cfg["sliced"]["seed"], self.cfg["sliced"]["orthogonal_bases"]
        )
        self.sensitivity_distance = SlicedDistance(
            self.budget,
            self.cfg["sliced"]["sensitivity_seed"],
            self.cfg["sliced"]["orthogonal_bases"],
        )
        self.marginal_distance = SlicedDistance(self.budget, 0, axes=True)
        self.analysis_distance = SlicedDistance(
            self.budget, self.adaptive_cfg["transport_direction_seed"]
        )
        self.references = read(folder / "adaptive_references.json")
        self.features = read(folder / "features.json")
        mapping = asset_root / dataset / "mapping"
        self.selection = read(mapping / "selection.json")
        self.axes = self.selection["axes"]
        assert self.selection["pca_identity"] == self.calendar["pca_identity"]
        self.mapper = {
            k: torch.as_tensor(v, device="cuda", dtype=torch.float64)
            for k, v in dict(np.load(mapping / "model.npz")).items()
        }
        self.cloud_cache = {}
        self.target_cache = {}
        self.gpu_observation_cache = {}
        self.semantic_cache = {}
        self.additional_target_cache = {}
        from traser.utils.resources import ContainerGuard

        self.guard = ContainerGuard(self.cfg["resources"])
        self.guard()
        self.identity = {
            "release_replay": True,
            "dataset": dataset,
            "window_hours": hours,
            "particles": self.budget,
            "pca_identity": self.calendar["pca_identity"],
            "mapping_sha256": sha(mapping / "model.npz"),
        }
        self.analysis_identity = self.identity
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def initial(self, phase, seed, origin=None):
        from traser.dynamics.base import Experiment as Base

        return Base.initial(self, phase, seed, origin)

    def progress(self, state, **kw):
        print(
            json.dumps({"stage": state, "dataset": self.dataset, "hours": self.window_hours, **kw}),
            flush=True,
        )


def requirements(dataset, root):
    cfg = read(ROOT / f"configs/experiments/{('blm' if dataset == 'blm_x' else 'weibo')}.json")
    required = []
    for name in [
        "representation/coordinates.parquet",
        "mapping/model.npz",
        "mapping/selection.json",
    ]:
        required.append(root / dataset / name)
    for h in cfg["window_hours"]:
        required.extend(
            (
                root / dataset / f"w{h}" / name
                for name in [
                    "calendar.json",
                    "observations.parquet",
                    "selection.json",
                    "adaptive_references.json",
                    "features.json",
                ]
            )
        )
    return (cfg, required)


@exclusive_experiment
def run(dataset, asset_root=None, check_only=False, references=False, particles=None):
    root = (asset_root or ROOT / "assets").resolve()
    cfg, paths = requirements(dataset, root)
    missing = [str(p.relative_to(root)) for p in paths if not p.exists()]
    result = {
        "dataset": dataset,
        "asset_root": str(root),
        "missing_assets": missing,
        "complete_asset_tree": not missing,
        "scientific_run_executed": False,
    }
    if check_only:
        print(json.dumps(result, indent=2))
        return result
    if missing:
        raise FileNotFoundError(
            "Frozen replay assets missing: "
            + ", ".join(missing)
            + ". See docs/REPRODUCIBILITY.md. No substitute sampled run is performed."
        )
    from traser.experiments.chains import chain

    allrows = []
    for h in cfg["window_hours"]:
        exp = Experiment(dataset, h, root, particles)
        selected = read(root / dataset / f"w{h}/selection.json")
        for phase in ["train", "development", "test"]:
            for seed in cfg["seeds"]:
                for group in ["H0", "HF", "HA"]:
                    spec = (
                        selected["selected"]["spec"] if group == "HA" else selected["fixed"]["spec"]
                    )
                    value = chain(
                        exp,
                        phase,
                        group,
                        seed,
                        selected["bcm"],
                        spec,
                        exp.out / f"{phase}/{group}/seed_{seed}",
                        diagnostics=True,
                    )
                    allrows.extend(value["rows"])
        if references and h == 12:
            exp.out = ROOT / f"outputs/replay/{dataset}/references"
            exp.out.mkdir(parents=True, exist_ok=True)
            chosen = exp.select_references()
            for name in ["WAR", "WR", "LQD"]:
                model = exp.torch.load(
                    exp.out / f"references/candidate_{chosen[name]['index']}/model.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                for seed in cfg["seeds"]:
                    folder = exp.out / f"test/{name}/seed_{seed}"
                    result = exp.reference_forecasts(folder, model, name, "test", [1], seed)
                    metrics = []
                    for row in result["pairs"]:
                        prediction = np.load(folder / row["prediction"])
                        evaluated = exp.metrics(prediction, row["target"], row)
                        if any(
                            (
                                r["reason"]
                                not in [
                                    "KDE undefined with fewer than 2 occurrences",
                                    "Degenerate marginal KDE bandwidth",
                                ]
                                for r in evaluated["density_failures"]
                            )
                        ):
                            raise RuntimeError("Numerical reference density failure")
                        density = {
                            "symmetric_KL",
                            "sqrt_JS_arithmetic",
                            "sqrt_JS_geometric_unnormalized",
                            "density_L1",
                        }
                        assert all(
                            (
                                v is None and k in density or (v is not None and np.isfinite(v))
                                for k, v in evaluated["metrics"].items()
                            )
                        )
                        metrics.append(
                            {
                                **row,
                                "dataset": dataset,
                                "window_hours": h,
                                "method": name,
                                **evaluated["metrics"],
                                "density_failures": evaluated["density_failures"],
                            }
                        )
                    write(folder / "all_metrics.json", metrics)
        exp.clear_phase()
    suffix = "" if particles is None else f"_N{particles}"
    write(ROOT / f"outputs/replay/{dataset}/forecast_rows{suffix}.json", allrows)
    return allrows
